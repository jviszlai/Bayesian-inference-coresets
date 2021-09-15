"""
A set of functions for solving the k-means clustering problem using QAOA
"""
import sys, os
import pickle, datetime, math, sympy
import random as rand
import numpy as np
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from copy import copy
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.tools.monitor import job_monitor
from qiskit.tools.visualization import plot_histogram
from active_coresets.optimizers.Nelder_Mead import minimizeEnergyObjective
from active_coresets.optimizers.Pennylane_Gradient_Descent import gradient_descent
from scipy.optimize import minimize


def convert_to_H(term_to_coef):
    H = []
    for term in term_to_coef.keys():
        coef = term_to_coef[term]
        pauli_str = ['Z{}'.format(i) for i in range(len(term)) if term[i] == 1]
        H.append((coef, pauli_str))
    return H

def graph_from_data(data_points):
    G = nx.Graph()
    G.add_nodes_from(list(range(len(data_points))))

    node_to_point = {}
    for node in range(len(data_points)):
        node_to_point[node] = data_points[node]

    for v1 in range(len(data_points)-1):
        for v2 in range(v1+1, len(data_points)):
            G.add_edge(v1, v2)
            G[v1][v2]['weight'] = np.linalg.norm(np.array(data_points[v1]) - np.array(data_points[v2]))

    return G, node_to_point

'''
def hamiltonian_from_graph(G):
    "Convert a networkx graph into a Hamiltonian"
    H = []
    for edge in G.edges():
        v1, v2 = edge
        coef = G[v1][v2]['weight']
        # Graph nodes [0,1,...,n] map to qubits [0,1,...,n]
        pauli_str = ['Z{}'.format(v) for v in [v1, v2]]
        H.append((coef, pauli_str))
    return H
'''

def gen_sympy_hamiltonian(n, coreset, taylor_order=0, r_plus=0.5):
    # r_minus is determined by r_plus
    if r_plus < 0 or r_plus > 1:
        raise Exception('invalid value for r_plus = {}. Must be in [0,1]'.format(r_plus))
    r_minus = 1 - r_plus

    Z = sympy.symbols('Z0:{}'.format(n))
    # coreset is a weighted list of points
    # [(weight, np.array), ...]
    w = [point[0] for point in coreset]
    x = [point[1] for point in coreset]
    W = sum(w)

    # create the weight ratios as symbols
    W_plusminus = sympy.symbols('W_pm')
    W_minusplus = sympy.symbols('W_mp')

    H_sym = 0
    # first term
    for i in range(n):
        coef = (1 - Z[i])/2 * W_plusminus + (1 + Z[i])/2 * W_minusplus
        H_sym += coef * w[i]**2 * np.dot(x[i], x[i])
    # second term
    for i in range(n):
        for j in range(i+1, n):
            coef = (2 + Z[i] + Z[j])/4 * W_minusplus + (2 - Z[i] - Z[j])/4 * W_plusminus - (1 - Z[i]*Z[j])/4 * (W_plusminus + W_minusplus + 2)
            H_sym += 2 * coef * w[i] * w[j] * np.dot(x[i],x[j])

    H_expanded = sympy.expand(H_sym)

    if taylor_order == 0:
        H_expanded = H_expanded.subs(W_plusminus, (1/r_minus) - 1)
        H_expanded = H_expanded.subs(W_minusplus, (1/r_plus) - 1)
    elif taylor_order == 1:
        fullsum = 0
        for l in range(n):
            fullsum += w[l]*Z[l]
        H_expanded = H_expanded.subs(W_plusminus, 2/r_minus - 1 - 1/(2*r_minus**2) + 1/(2*W*r_minus**2) * fullsum)
        H_expanded = H_expanded.subs(W_minusplus, 2/r_plus - 1 - 1/(2*r_plus**2) - 1/(2*W*r_plus**2) * fullsum)
        H_expanded = sympy.simplify(sympy.expand(H_expanded))
    else:
        raise Exception('Taylor order expansion > 1 not supported')

    term_to_coef = {}
    for term in H_expanded.as_terms()[0]:
        coefficient = term[1][0][0]
        variable_tuple = term[1][1]
        # for Pauli operators, Z^2 = I
        variable_tuple = tuple([x % 2 for x in variable_tuple])
        if any(variable_tuple): # skip constant terms, i.e. the (0,0,...,0) tuple
            term_to_coef[variable_tuple] = coefficient

    return term_to_coef


def gen_coreset_graph(coreset=4, taylor_order=0, r_plus=0.5, metric='dot'):
    """
    Generate a complete weighted graph using the provided set of coreset points

    Parameters
    ----------
    coreset : List((weight, vector)) or int
        Set of coreset points to use. Each point should consist of a weight
        value and a numpy array as the vector
    taylor_order : int
        Order of the Taylor expansion to use when generating the weights
    r_plus : float
        The ratio between one of the cluster sizes to the total data size.
        The r_minus ratio is then r_minus = 1 - r_plus
    metric : str
        Choose the desired metric for computing the edge weights.
        Options include: dot, dist

    Returns
    -------
    coreset : List((weight, vector))
        The set of points used to construct the graph
    G : NetworkX Graph
        A complete weighted graph
    H : List((coef, pauli_string))
        The equivalent Hamiltonian for the generated graph
    """
    if type(coreset) is int:
        num_points = coreset
        # Generate a graph instance with sample coreset data
        coreset = []
        cluster1_size = num_points // 2
        # generate points around x=-1, y=-1
        for _ in range(cluster1_size):
            # use a uniformly random weight
            #weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array([np.random.normal(loc=-1, scale=0.5, size=1)[0],
                               np.random.normal(loc=-1, scale=0.5, size=1)[0]])
            new_point = (weight, vector)
            coreset.append(new_point)

        cluster2_size = num_points - cluster1_size
        # generate points around x=+1, y=1
        for _ in range(cluster2_size):
            # use a uniformly random weight
            #weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array([np.random.normal(loc=1, scale=0.5, size=1)[0],
                               np.random.normal(loc=1, scale=0.5, size=1)[0]])
            new_point = (weight, vector)
            coreset.append(new_point)
    else:
        # check that the provided coreset is formatted properly
        if type(coreset[0][1]) is float or type(coreset[0][1]) is np.float64:
            raise Exception('Incorrect coreset format. Make sure weights are included')

    # Generate a networkx graph with correct edge weights
    n = len(coreset)
    G = nx.complete_graph(n)
    H = []

    # use sympy to generate the corresponding Hamiltonian
    term_to_coef = gen_sympy_hamiltonian(n, coreset, taylor_order=taylor_order, r_plus=r_plus)

    # Properly weight the graph and construct the corresponding Hamiltonian
    for edge in G.edges():
        pauli_str = ['I']*n
        idx_tuple = [0]*n
        # coreset points are labelled by their vertex index
        v_i = edge[0]
        v_j = edge[1]
        # The pauli strings are indexed in BIG ENDIAN fahsion
        # i.e. ->   ZZII = Z_0.Z_1.I_2.I_3
        pauli_str[v_i] = 'Z'
        pauli_str[v_j] = 'Z'
        idx_tuple[v_i] = 1
        idx_tuple[v_j] = 1
        weight = term_to_coef[tuple(idx_tuple)]
        G[v_i][v_j]['weight'] = weight
        H.append((weight, pauli_str))

    return coreset, G, H


def plot_coreset_graph(coreset_points, G, twoD=True, centroids=None):
    """
    Plot the coreset points on the x-y plane, and draw the networkX graph
    """
    # Plot the coreset points
    if twoD:
        xx = [cp[1][0] for cp in coreset_points]
        yy = [cp[1][1] for cp in coreset_points]
        plt.scatter(xx, yy)
        plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
        plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')

        if not centroids is None:
            for point in centroids:
                plt.scatter(point[0], point[1], marker='*', c='r', s=80)

        plt.show()
        plt.close()

    if not G is None:
        # Generate plot of the Graph
        colors = ['r' for node in G.nodes()]
        default_axes = plt.axes(frameon=False)
        pos = nx.spring_layout(G)

        nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1,
                         ax=default_axes, pos=pos)


# Create the quantum circuit to implement the QAOA
def evolve_cost(c, angle, G, ising=False, qbit_map=None):
    """
    Implement the evolution according to the Cost Hamiltonian

    Parameters
    ----------
    c : QuantumCircuit
        The circuit instance to add gates to
    angle : float
        The current value of gamma
    G : NetworkX graph
        The graph for the current problem instance
    ising : bool
        If false, use CNOTs. If true, use controlled-u1 gates
    qbit_map : List
        If set to None, then the circuit will be constructed assuming an
        all-to-all connectivity. Otherwise, passing in a QPU connectivity list
        will result in a linear implementation of the QAOA relying on the
        fermionic swap networks discussed in (https://arxiv.org/abs/1711.04789)

    Returns
    -------
    Nothing, all gates added inplace
    """
    if qbit_map is None:
        for edge in G.edges():
            # Graph nodes [0,1,...,n] map to qubits [0,1,...,n]
            i = edge[0]
            j = edge[1]
            phi = angle * G[i][j]['weight']
            if ising:
                c.cu1(-4*phi, i, j)
                c.u1(2*phi, i)
                c.u1(2*phi, j)
            else:
                c.cx(i,j)
                c.rz(2*phi, j)
                c.cx(i,j)
    else: # implement the swap network
        # The covers indicate which qubits should be swapped at each layer
        cover_a = [(idx-1, idx) for idx in range(1,len(qbit_map),2)]
        cover_b = [(idx-1, idx) for idx in range(2,len(qbit_map),2)]

        # We also need to keep track of where the "virtual" qubits are so
        # that we use the correct edge weight when computing phi
        # virtual_map -> indices indicate physical qubit, values indicate
        # the virtual qubit residing there.
        virtual_map = np.arange(len(qbit_map))

        for l, layer in enumerate(range(len(G.nodes))):
            cover = [cover_a, cover_b][layer % 2]
            for pair in cover:
                i, j = pair # swap physical qubits i and j
                # get the edge weight using the virtual_map
                v_i = virtual_map[i]
                v_j = virtual_map[j]
                phi = angle * G[v_i][v_j]['weight']
                c.cx(i, j)
                c.rz(2*phi, j)
                if l == len(G.nodes) - 1:
                    c.cx(i, j) # the final SWAP doesn't need to happen
                else:
                    c.cx(j, i)
                    c.cx(i, j)
                # update the virtual_map with the current SWAP
                virtual_map[j], virtual_map[i] = virtual_map[i], virtual_map[j]


def evolve_driver(c, angle):
    """
    Implement the evolution according to the Driver Hamiltonian

    Parameters
    ----------
    c : QuantumCircuit
        The circuit instance to add gates to
    angle : float
        The current value of beta

    Returns
    -------
    Nothing, all gates added inplace
    """
    c.rx(2*angle, c.qubits)


def gen_complete_qaoa_circ(P, gamma, beta, G, ising=False, topology=[],
                           measure=True):
    """

    Parameters
    ----------
    P : int
        number of layers to apply
    gamma : List[float]
        list with length p, contains angle parameters for cost Hamiltonian
    beta : List[float]
        list with length p, contains angle parameters for driver Hamiltonian
    G : NetworkX graph
        graph representing the MAXCUT problem instance
    ising : bool
        If true, use Cu1 gates to implement cost evolution, otherwise use cx
    topology : List
        If set to empty list, then the circuit will be constructed assuming an
        all-to-all connectivity. Otherwise, passing in a QPU connectivity list
        will result in a linear implementation of the QAOA relying on the
        fermionic swap networks discussed in (https://arxiv.org/abs/1711.04789)
    measure : bool
        If true, append measurements to all qubits at the end of the circuit.
        If false, do not append measurements.

    Returns
    -------
    The QuantumCircuit implementing MAXCUT QAOA

        or, if a device topology is given

    The QuantumCircuit and the map from virtual to physical qubits
    """
    # create the quantum and classical registers
    n = len(G.nodes()) # get the number of vertices (qubits)
    circ = QuantumCircuit(n)

    # apply the initial layer of Hadamards
    circ.h(range(n))

    # iteratively apply the cost and driver unitaries p times
    if len(topology) == 0:
        for p in range(P):
            circ.barrier()
            evolve_cost(circ, gamma[p], G, ising, None)
            circ.barrier()
            evolve_driver(circ, beta[p])
    else:
        # implement the qaoa circuit using a swap network to minimize CNOT count
        # for now, assume a linear mapping
        initial_layout = np.arange(0,n)

        for p in range(P):
            circ.barrier()
            evolve_cost(circ, gamma[p], G, ising, initial_layout)
            circ.barrier()
            evolve_driver(circ, beta[p])

    if measure:
        # apply measurements to all qubits
        circ.measure_all()

    if len(topology) == 0:
        return circ, None
    else:
        return circ, initial_layout


def qaoa_circ_from_H(n, H, P, gammas, betas):
    c = QuantumCircuit(n)

    # apply the initial layer of Hadamards
    c.h(range(n))

    # iteratively apply the cost and driver unitaries P times
    for p in range(P):
        c.barrier()

        # evolve under cost Hamiltonian
        for term in H:
            coef = term[0]
            qubit_idxs = [int(pstr[1:]) for pstr in term[1]]

            # CNOT ladder - compute
            for ctrl_idx in range(len(qubit_idxs) - 1):
                trgt_idx = ctrl_idx + 1
                c.cx(qubit_idxs[ctrl_idx], qubit_idxs[trgt_idx])
            # RZ rotation
            c.rz(2*gammas[p]*coef, qubit_idxs[-1])
            # CNOT ladder - uncompute
            for ctrl_idx in reversed(range(len(qubit_idxs) - 1)):
                trgt_idx = ctrl_idx + 1
                c.cx(qubit_idxs[ctrl_idx], qubit_idxs[trgt_idx])

        c.barrier()

        # evolve under driver Hamiltonian
        c.rx(2*betas[p], c.qubits)

    return c


def cost_function_C(x, G):
    """
    Compute the value of the cost function.

    This function assumes a mapping between vertices and qubits:

            v0, v1, ..., vN -> q0, q1, ..., qN

    Parameters
    ----------
    x : List[int]
        Bitstring corresponding to a MAXCUT partitioning
    G : NetworkX Graph
        The current MAXCUT problem instance

    Returns
    -------
    float : the value of the cost function evaluated for the given x and G
    """
    E = G.edges()
    if( len(x) != len(G.nodes())):
        return np.nan
    if type(x) is str:
        x = list(x)
        x = [int(xx) for xx in x]

    C = 0
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]['weight']
        # Equation 6 in Overleaf Paper
        C = C + w*(1 - 2*(x[e1]*(1-x[e2]) + x[e2]*(1-x[e1])))
        #C = C - w * (x[e1]*(1-x[e2]) + x[e2]*(1-x[e1]))

    return C

def maxcut_energy_from_bitstring(x, G):
    # convert 0,1 bitstring to 1,-1 bitstring
    bitstr = []
    for val in x:
        if val == 0:
            bitstr.append(1)
        elif val == 1:
            bitstr.append(-1)
    # reverse the order so we can use list indexing to select the right qubit
    # Example: bitstr = 0101, qubit0 = 1, bitstr[0] = 0
    bitstr = list(reversed(bitstr))

    eng = 0
    for edge in G.edges():
        v1, v2 = edge
        eng += G[v1][v2]['weight'] * (1 - bitstr[v1]*bitstr[v2]) / 2

    return eng


def energy_from_bitstring(x, H):
    # convert 0,1 bitstring to 1,-1 bitstring
    bitstr = []
    for val in x:
        if val == 0:
            bitstr.append(1)
        elif val == 1:
            bitstr.append(-1)
    # reverse the order so we can use list indexing to select the right qubit
    # Example: bitstr = 0101, qubit0 = 1, bitstr[0] = 0
    bitstr = list(reversed(bitstr))

    eng = 0
    for term in H:
        eng += term[0]*np.prod([bitstr[int(op[1:])] for op in term[1]])

    return eng


def reorder_bitstrings(P, nq, old_counts):
    """
    If the swap network is used to implement the cost evolution, then the
    measured bitstrings need to be reordered to account for the last SWAP layer
    that is removed in the network.

    Qiskit orders the measurement bitstrings little endian: qN,...,q1,q0
    This function applies the last SWAP layer of the network and then performs
    one final reversal so that the initial mapping between vertices and qubits:
            v0,v1,...,vN -> q0,q1,...,qN
    is preserved in the final bitstring counts:
            {q0q1...qN: 1024, q0q1...qN:2003, ...}
    """
    assert (P == 1),'Reordering only implemented for P=1'
    cover_a = [(idx-1, idx) for idx in range(1,nq,2)]
    cover_b = [(idx-1, idx) for idx in range(2,nq,2)]
    last_cover = [cover_b, cover_a][nq % 2]
    new_counts = {}
    for bitstr in old_counts.keys():
        bit_list = list(bitstr) # convert the bitstr to a list
        bit_list.reverse() # Qiskit orders the qubit bitstring little endian
        for i, j in last_cover:
            bit_list[j], bit_list[i] = bit_list[i], bit_list[j]
        bit_list.reverse() # reverse the final layer of the swap network
        new_bitstr = ''.join(bit_list)
        new_counts[new_bitstr] = old_counts[bitstr]
    return new_counts


def energy_landscape(P, step_size, shots, gammaLim, betaLim, G,
                     device_topology=[], device=None):
    """
    Use the QASM simulator to generate the energy landscape
    """
    backend = Aer.get_backend('qasm_simulator')
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    if device is None:
        reorder = False
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots)
    else:
        assert isinstance(device, IBMQBackend), 'device must be an IBMQBackend'
        reorder = True
        noise_model = NoiseModel.from_backend(device)
        basis_gates = device.configuration().basis_gates
        coupling_map= device.configuration().coupling_map
        props = device.properties()
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots,
                    basis_gates=basis_gates, noise_model=noise_model,
                    coupling_map=coupling_map, backend_properties=props)

    estC = []
    bitstrings = {}
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        # scan across the gamma-beta plane
        C_row = []
        for gamma, beta in zip(gamma_list, beta_list):
            # TODO: this is still somewhat of a hack, because only one
            # value of gamma and beta can be passed to the circuit at a single time
            output = gen_complete_qaoa_circ(P, [gamma], [beta], G,
                                            topology=device_topology)

            if len(device_topology) == 0:
                qaoa_circ = output
            else:
                qaoa_circ, initial_layout = output

            # Simulate, either noisy or noiseless
            simulate = execute_func(qaoa_circ, backend, shots)
            raw_counts = simulate.result().get_counts()

            # reorder the outputs if the swap network was used
            if reorder is True:
                counts = reorder_bitstrings(P, len(G.nodes), copy(raw_counts))
            else:
                counts = raw_counts

            # Save the counts, indexed by the current gamma and beta values
            bitstrings['{:.3f}{:.3f}'.format(gamma, beta)] = counts

            # Evaluate the data from the simulator
            tot_C = 0
            for sample in list(counts.keys()):
                # use the sampled bitstring x to compute C(x)
                x = [int(num) for num in list(sample)]
                tmp_eng = cost_function_C(x,G)

                # compute the expectation value
                tot_C = tot_C + counts[sample]*tmp_eng

            # average the cost function over the number of shots
            avg_C = tot_C/shots
            C_row.append(avg_C)

        # save the entire row of avg_C values
        estC.append(C_row)

    # convert estC to a numpy array
    estC = np.array(estC)
    return estC, bitstrings


def plot_energy_landscape(step_size, gammaLim, betaLim, estC, bitstrings, shots,
                          coreset_points, G, savefigs=None):
    # Plot the energy landscape
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    fig = plt.figure()
    ax  = fig.gca(projection='3d')

    surf = ax.plot_surface(a_gamma, a_beta, estC, cmap=cm.coolwarm, linewidth=0,
                           antialiased=True)

    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel('Cost function')
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.tight_layout()
    if savefigs is not None: plt.savefig('qaoa_energylandscape_{}.pdf'.format(savefigs))
    plt.show()
    plt.close()

    result = np.where(estC == np.amax(estC))
    a = list(zip(result[0],result[1]))[0]

    gamma  = a[0]*step_size;
    beta   = a[1]*step_size;
    optimal_counts = bitstrings['{:.3f}{:.3f}'.format(gamma, beta)]
    bs_list = [(key, optimal_counts[key]) for key in optimal_counts.keys()]
    bs_list = sorted(bs_list, key=lambda tup: tup[1], reverse=True)

    #The smallest parameters and the expectation can be extracted
    print('\n --- OPTIMAL PARAMETERS --- \n')
    print('The maximal expectation value (avg over {} shots) is:  C = {:.3f}'.format(shots, np.amax(estC)))
    print('This is attained for gamma = {0:.3f} and beta = {1:.3f}'.format(gamma,beta))
    print('The 4 most common partitionings produced at this point were:')
    print('{}: {:.2f}% ({}), {}: {:.2f}% ({}), {}: {:.2f}% ({}), {}: {:.2f}% ({})'.format(
       bs_list[0][0], 100*bs_list[0][1]/shots, cost_function_C(bs_list[0][0], G),
       bs_list[1][0], 100*bs_list[1][1]/shots, cost_function_C(bs_list[1][0], G),
       bs_list[2][0], 100*bs_list[2][1]/shots, cost_function_C(bs_list[2][0], G),
       bs_list[3][0], 100*bs_list[3][1]/shots, cost_function_C(bs_list[3][0], G)))

    # Plot the centroids according to the optimal partitioning
    opt_partition = bs_list[0][0]
    c_plus = 'red'
    c_minus = 'blue'
    xx = [cp[1][0] for cp in coreset_points]
    yy = [cp[1][1] for cp in coreset_points]

    S_plus = [coreset_points[i] for i in range(len(opt_partition)) if opt_partition[i] == '1']
    xx_plus = [pt[1][0] for pt in S_plus]
    yy_plus = [pt[1][1] for pt in S_plus]
    plt.scatter(xx_plus, yy_plus, c=c_plus)

    S_minus = [coreset_points[i] for i in range(len(opt_partition)) if opt_partition[i] == '0']
    xx_minus = [pt[1][0] for pt in S_minus]
    yy_minus = [pt[1][1] for pt in S_minus]
    plt.scatter(xx_minus, yy_minus, c=c_minus)

    mu_plus = np.sum([point[0]*point[1] for point in S_plus], axis=0) / np.sum([point[0] for point in S_plus])
    mu_minus = np.sum([point[0]*point[1] for point in S_minus], axis=0) / np.sum([point[0] for point in S_minus])
    print('mu_plus:',mu_plus)
    print('mu_minus:',mu_minus)
    try:
        plt.scatter(mu_plus[0], mu_plus[1], c=c_plus, marker='*')
    except:
        mu_plus = None
    try:
        plt.scatter(mu_minus[0], mu_minus[1], c=c_minus, marker='*')
    except:
        mu_minus = None

    plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
    plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')
    if savefigs is not None: plt.savefig('qaoa_clustering_{}.pdf'.format(savefigs))
    plt.show()
    plt.close()

    # Plot the histogram of measurement counts at the optimal gamma-beta pair
    return optimal_counts


def plot_partition(bitstring, coreset_points):
    c_plus = 'blue'
    c_minus = 'red'

    xx = [cp[1][0] for cp in coreset_points]
    yy = [cp[1][1] for cp in coreset_points]

    S_plus = [coreset_points[i] for i in range(len(bitstring)) if bitstring[i] == '1']
    xx_plus = [pt[1][0] for pt in S_plus]
    yy_plus = [pt[1][1] for pt in S_plus]
    plt.scatter(xx_plus, yy_plus, c=c_plus)

    S_minus = [coreset_points[i] for i in range(len(bitstring)) if bitstring[i] == '0']
    xx_minus = [pt[1][0] for pt in S_minus]
    yy_minus = [pt[1][1] for pt in S_minus]
    plt.scatter(xx_minus, yy_minus, c=c_minus)

    mu_plus = np.sum([point[0]*point[1] for point in S_plus], axis=0) / np.sum([point[0] for point in S_plus])
    mu_minus = np.sum([point[0]*point[1] for point in S_minus], axis=0) / np.sum([point[0] for point in S_minus])
    print('mu_plus:',mu_plus)
    print('mu_minus:',mu_minus)
    plt.scatter(mu_plus[0], mu_plus[1], c=c_plus, marker='*')
    plt.scatter(mu_minus[0], mu_minus[1], c=c_minus, marker='*')

    plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
    plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')
    plt.show()
    plt.close()


def hardware_execution(device, P, gammaLim, betaLim, step_size, G, shots):
    """
    Run on a quantum processor
    """
    # create a savename based on time of execution
    d = datetime.datetime.today()
    savename = device.name()+'_'+d.strftime('%m-%d-%Y-%H%M')
    savedir  = 'HardwareRunPickles/'
    print('Running {}'.format(savename))

    # collect all gamma-beta pairs
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    angle_pairs = []
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        for gamma, beta in zip(gamma_list, beta_list):
            angle_pairs.append((gamma, beta))

    # generate a qaoa circuit for each angle pair
    circuits = []
    all_angle_pairs = {}
    for gamma, beta in angle_pairs:
        circ, initial_layout = gen_complete_qaoa_circ(P, [gamma], [beta], G,
                                   topology=device.configuration().coupling_map)
        circuits.append(circ)
        # store the gamma-beta pair and the circuit name together
        all_angle_pairs[circ.name] = (gamma, beta)
    print('Generated {} circuits to cover gamma:{} -> {:.2f}, beta:{} -> {:.2f}'.format(
           len(circuits), *gammaLim, *betaLim))

    # get the max number of experiments-per-job for this device
    maxExps = device.configuration().to_dict()['max_experiments']
    print('Device {} supports {} experiments per job'.format(device.name(), maxExps))

    # break the list of circuits into chunks of this size
    batches = [circuits[i:i + maxExps] for i in range(0, len(circuits), maxExps)]
    print('Batched all {} circuits into {} total jobs'.format(len(circuits), len(batches)))

    # execute the jobs
    print('LAUNCHING JOBS')
    all_raw_counts = {}
    for num, batch in enumerate(batches):
        print('Executing job {}/{}'.format(num+1, len(batches)))
        job = execute(batch, backend=device, shots=shots,
                      initial_layout=initial_layout)
        job_monitor(job)
        for experiment in batch:
            raw_counts = job.result().get_counts(experiment=experiment)
            all_raw_counts[experiment.name] = raw_counts

    # pickle the raw counts
    pklFile = open(savedir+savename+'_rawcounts.pickle', 'wb')
    pickle.dump(all_raw_counts, pklFile)
    pklFile.close()
    print('raw counts saved')

    # reorder the raw counts
    good_counts = {}
    for circkey in all_raw_counts.keys():
        old_counts = all_raw_counts[circkey]
        new_counts = reorder_bitstrings(P, len(G.nodes), old_counts)
        good_counts[circkey] = new_counts

    # match the gamma-beta pairs with the execution counts
    execution_dict = {}
    for circkey in all_angle_pairs.keys():
        gamma, beta = all_angle_pairs[circkey]
        execution_dict['{:.3f}{:.3f}'.format(gamma, beta)] = good_counts[circkey]

    # pickle the execution dictionary
    pklFile = open(savedir+savename+'_fullrun.pickle', 'wb')
    pickle.dump(execution_dict, pklFile)
    pklFile.close()
    print('execution dictionary saved')

    return execution_dict


def load_fullrun(picklename):
    pklFile = open(picklename, 'rb')
    fullrun = pickle.load(pklFile)
    pklFile.close()
    return fullrun


def compute_C_from_fullrun(fullrun, gammaLim, betaLim, step_size, G, shots):
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    estC = []
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        C_row = []
        for gamma, beta in zip(gamma_list, beta_list):
            counts = fullrun['{:.3f}{:.3f}'.format(gamma, beta)]

            # Evaluate the cost function
            tot_C = 0
            for sample in list(counts.keys()):
                # use the sampled bitstring x to compute C(x)
                x = [int(num) for num in list(sample)]
                tmp_eng = cost_function_C(x,G)

                # compute the expectation value
                tot_C = tot_C + counts[sample]*tmp_eng

            # average the cost function over the number of shots
            avg_C = tot_C/shots
            C_row.append(avg_C)

        # save the entire row of avg_C values
        estC.append(C_row)

    # convert estC to a numpy array
    estC = np.array(estC)
    return estC


def sim_and_evaluate(circ, P, G, shots, reorder, device=None, qaoa_opts=None):
    """
    Simulate the circuit using either statevector or shot-based simulation, then
    compute the expected value of the cost function.
    """
    if shots is None and device is None:
        backend = Aer.get_backend('statevector_simulator')
    else:
        if shots is None:
            shots = 8192 #override shots with default value
        backend = Aer.get_backend('qasm_simulator')

    # Have the option of conducting noisy simulation
    if device is None:
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots)
    else:
        assert isinstance(device, IBMQBackend), 'device must be an IBMQBackend'
        noise_model = NoiseModel.from_backend(device)
        basis_gates = device.configuration().basis_gates
        coupling_map= device.configuration().coupling_map
        props = device.properties()
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots,
                    basis_gates=basis_gates, noise_model=noise_model,
                    coupling_map=coupling_map, backend_properties=props)

    # execute the circuit
    simulate = execute_func(circ, backend, shots)
    result = simulate.result()

    if shots is None:
        # statevector simulation: get the probabilities
        statevector = Statevector(result.get_statevector(circ))
        raw_counts = statevector.probabilities_dict()
        shots = 1.0 # denominator is 1.0 since probability vector
    else:
        # shot-based simulation: get the counts
        raw_counts = result.get_counts(circ)

    # reorder the outputs if the swap network was used
    if reorder is True:
        counts = reorder_bitstrings(P, len(G.nodes), copy(raw_counts))
    else:
        counts = raw_counts

    # Evaluate the data from the simulator
    if qaoa_opts is None or qaoa_opts['gibbs'] is False:
        # Compute the energy objective function
        tot_C = 0
        for sample in counts.keys():
            # use the sampled bitstring x to compute C(x)
            x = [int(num) for num in list(sample)]
            tmp_eng = cost_function_C(x,G)

            # compute the expectation value
            tot_C = tot_C + counts[sample]*tmp_eng

        # average the cost function over the number of shots
        avg_C = tot_C/shots
        return avg_C
    elif qaoa_opts['gibbs'] is True:
        # Optimize over the Gibbs objective function
        keys = list(qaoa_opts.keys())
        if not 'eta' in keys: raise Exception('key: \'eta\' must be provided in qaoa_opts')
        eta = qaoa_opts['eta']
        gibbs_sum = 0
        for sample in counts.keys():
            # compute the energy for the given bitstring x
            x = [int(num) for num in list(sample)]
            energy = cost_function_C(x,G)
            gibbs_sum += gibbs_sum + counts[sample]*(np.exp(-eta*energy))
        # average over the number of shots and take the log
        return np.log(gibbs_sum/shots)


def optimize_qaoa(data_vectors, P=1, taylor_order=0, r_plus=0.5,
                  delta=1.7, tol=10e-6, no_improv_break=15, max_iter=2000, print_info=2):

    term_to_coef = gen_sympy_hamiltonian(len(data_vectors), data_vectors,
                                            taylor_order=taylor_order, r_plus=r_plus)
    H = convert_to_H(term_to_coef)

    backend = Aer.get_backend('statevector_simulator')

    def f(params):
        gammas = [y for i, y in enumerate(params) if i % 2 == 0]
        betas  = [b for i, b in enumerate(params) if i % 2 == 1]

        circ_f = qaoa_circ_from_H(len(data_vectors), H, P, gammas, betas)

        # Compute the cost function
        result = execute(circ_f, backend=backend).result()
        statevector = Statevector(result.get_statevector(circ_f))
        counts = statevector.probabilities_dict()

        avg_cost = 0
        for sample in counts.keys():
            x = [int(num) for num in list(sample)]
            avg_cost += counts[sample] * energy_from_bitstring(x, H)

        return -avg_cost

    opt_params, opt_cost = minimizeEnergyObjective(f, None, 2*P, delta, tol, no_improv_break, max_iter, print_info)

    gammas = [y for i, y in enumerate(opt_params) if i % 2 == 0]
    betas  = [b for i, b in enumerate(opt_params) if i % 2 == 1]
    opt_circ = qaoa_circ_from_H(len(data_vectors), H, P, gammas, betas)

    if print_info >= 1:
        print('Best parameters:')
        print('\tGammas:', gammas)
        print('\tBetas:', betas)
        print('Best energy: {:.6f}'.format(-opt_cost))

    return opt_params, -opt_cost, opt_circ


def solve_maxcut(P, G, init_params=None, shots=None, topology=[], steps=100,
                 device=None, method='non-gradient', optimizer='nm', delta=1.7,
                 tol=10e-6, no_improv_break=15, max_iter=5000, print_info=2,
                 qaoa_opts=None, stepsize=0.1):
    """
    Find the optimal parameters to solve MAXCUT using QAOA on the given graph G

    Parameters
    ----------
    P : int
        The QAOA parameter controlling the number of driver/evolver layers
    G : NetworkX Graph
        The graph instance to solve MAXCUT on
    init_params : list[float]
        Seed the variational optimization with initial values for alpha and
        beta. Default is None which will select uniform random values
    shots : int
        If an int is provided, the qasm_simulator will be used with that number
        of shots. If None, then the statevector_simulator will be used.
    topology : list[(int,int),...]
        Pass in the device connectivity which will generate QAOA circuits using
        the fermionic SWAP network. Default is an empty list which will
        generate a QAOA circuit assuming all-to-all connectivity.
    device : IBMQBackend
        If an IBMQBackend is passed in, then noisy simulation will be used with
        the noisy model taken from the backend. Default is None which results in
        noiseless simulation. This parameter will override 'shots'.
    qaoa_opts : dict
        A dictionary of options to pass to the QAOA objective function.
        Options include {gibbs:True/False, eta:float}

    Returns
    -------
    List : values of the optimal parameters found by the optimizer
    float : the lowest value of the cost function found
    """

    backend = Aer.get_backend('statevector_simulator')

    def f(params):
        gamma = [y for i, y in enumerate(params) if i % 2 == 0]
        beta = [b for i, b in enumerate(params) if i % 2 == 1]

        # Generate a circuit for the qaoa
        circ_f, _ = gen_complete_qaoa_circ(P, gamma, beta, G, topology=topology,
                                           measure=False)

        # Compute the cost function
        result = execute(circ_f, backend=backend).result()
        statevector = Statevector(result.get_statevector(circ_f))
        counts = statevector.probabilities_dict()

        avg_cost = 0
        for sample in counts.keys():
            x = [int(num) for num in list(sample)]
            avg_cost += counts[sample] * maxcut_energy_from_bitstring(x, G)

        # the Nelder-Mead optimizer is minimizing the energy, so we return
        # the negative of the QAOA cost function
        return -avg_cost

    num_params = 2*P
    if method.lower() in ['grad', 'gradient', 'gradient_descent']:
        opt_params, opt_cost = gradient_descent(G, P, optimizer=optimizer,
                                                stepsize=stepsize, init_params=init_params,
                                                steps=steps)
    else:
        if optimizer.lower() in ['nm', 'nelder-mead', 'neldermead']:
            opt_params, opt_cost = minimizeEnergyObjective(f, init_params,
                                        num_params, delta, tol, no_improv_break,
                                        max_iter, print_info)
        elif optimizer.lower() in ['cobyla', 'powell']:
            if init_params is None:
                init_params = [rand.uniform(0,2*math.pi) for i in range(num_params)]
            out = minimize(f, x0=init_params, method=optimizer, options={'maxiter':500})
            opt_params = out['x']
            opt_cost = out['fun']
        else:
            raise Exception('Unknown optimization method: {}'.format(optimizer))

    if print_info >= 1:
        print('Best parameters:', opt_params)
        print('Best energy: {:.7f}'.format(opt_cost))

    opt_gamma = [y for i, y in enumerate(opt_params) if i % 2 == 0]
    opt_beta  = [b for i, b in enumerate(opt_params) if i % 2 == 1]
    opt_circ, _ = gen_complete_qaoa_circ(P, opt_gamma, opt_beta, G, measure=False)

    result = execute(opt_circ, backend).result()
    sv = Statevector(result.get_statevector(opt_circ))
    final_dist = sv.probabilities_dict()

    # select the most probable bitstring after the optimization
    max_prob = 0
    best_cut = None
    for sample in final_dist:
        if final_dist[sample] > max_prob:
            max_prob = final_dist[sample]
            best_cut = sample

    return best_cut


def compute_centroids(partition, coreset):
    S_plus = [coreset[i] for i in range(len(partition)) if partition[i] == '1']
    S_minus = [coreset[i] for i in range(len(partition)) if partition[i] == '0']
    mu_plus = np.sum([point[0]*point[1] for point in S_plus], axis=0) / np.sum([point[0] for point in S_plus])
    mu_minus = np.sum([point[0]*point[1] for point in S_minus], axis=0) / np.sum([point[0] for point in S_minus])
    #print('mu_plus:', mu_plus)
    #print('mu_minus:', mu_minus)
    return [mu_plus, mu_minus]


def gradient_based_optimization(init_params, P, G, shots, qaoa_opts):
    return


















