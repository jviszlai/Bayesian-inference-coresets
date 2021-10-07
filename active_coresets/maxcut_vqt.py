"""
A set of functions for solving the k-means clustering problem using VQT

Based on the original paper: https://arxiv.org/abs/1910.02071
And code found at: https://nbviewer.jupyter.org/github/Lucaman99/openqml/blob/master/notebooks/vqt.ipynb
"""
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import math
import itertools
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix


def create_density_plot(data):
    """
    Visualize a density matrix
    """
    array = np.array(data)
    plt.matshow(array)
    plt.colorbar()
    plt.show()
    plt.close()


def create_target(beta, Hamiltonian, plot=True, more_info=False):
    """
    Create the target density matrix

    Parameters
    ----------
    beta : float
        inverse temperature
    Hamiltonian : ndarray
        Matrix representing the MAXCUT Hamiltonian
    """

    y = -1 * float(beta) * Hamiltonian
    new_matrix = linalg.expm(np.array(y))
    norm = np.trace(new_matrix)
    final_target = (1/norm) * new_matrix

    # Calculate the entropy, expectation value, and final cost
    entropy = -1 * np.trace(np.matmul(final_target, linalg.logm(final_target)))
    ev = np.trace(np.matmul(final_target, Hamiltonian))
    real_cost = beta * np.trace(np.matmul(final_target, Hamiltonian)) - entropy

    # plot the density matrix
    if plot:
        create_density_plot(final_target.real)
    print('Entropy:', entropy)
    print('Expectation Value:', ev)
    print('Final Cost:', real_cost)

    if more_info:
        return DensityMatrix(final_target), entropy, ev, real_cost
    else:
        return DensityMatrix(final_target)


def create_maxcut_hamiltonian(H):
    """
    Generate the Hamiltonian matrix

    Parameters
    ----------
    H : list[(float, list[str])]
        The MAXCUT problem Hamiltonian as a list of tuples. Each tuple
        represents a term and contain the term's coefficient and the term's
        Pauli str

    Returns
    -------
    ndarray
    """
    pauli_z = np.array([[1, 0], [0, -1]])
    identity = np.array([[1, 0], [0, 1]])

    n = len(H[0][1]) # number of qubits
    matrix = np.zeros((2**n, 2**n))

    for term in H:
        coef, pauli_str = term
        # multiply the coefficient by -1 to turn the energy maximization
        # problem into an energy minimization problem
        m = -1*coef
        for pauli in pauli_str:
            if pauli == 'Z':
                m = np.kron(m, pauli_z)
            else:
                m = np.kron(m, identity)
        matrix = np.add(matrix, m)

    return matrix


def create_dist(theta):
    """
    Create the initial bitstring probability distribution parametrized by theta
    """
    prob = []
    for t in theta:
        prob.append([math.exp(-1*t), 1-math.exp(-1*t)])
    return prob


def cost_function(param, Hmatrix, beta, ansatz, print_info=2):
    """
    Evaluate the cost function for the given set of parameters
    """
    global counting_num

    nq = ansatz.nq

    # check the length of param
    # number of parameters = (1/2)*(n^2+5n)+n
    angles = ansatz.parse_params(param)
    theta = angles[-1]
    phi = angles[:-1]

    # Prepare the bitstring probability distribution
    opt_prob_dist = create_dist(theta)

    # Compute the probability
    p = []
    for elem in itertools.product(*opt_prob_dist):
        p.append(np.prod(elem))

    # Create empty initial density matrix
    state = np.zeros((2**nq, 2**nq))

    backend = Aer.get_backend('statevector_simulator')
    for l in range(2**nq):
        # Create the circuit
        circ = QuantumCircuit(nq)
        ansatz.construct_V(circ, [int(i) for i in list(bin(l)[2:].zfill(nq))])
        ansatz.construct_U(circ, *phi)

        # Simulate the circuit
        result = execute(circ, backend).result()
        outputstate = result.get_statevector(circ)

        # Adds a term to the "density matrix sum"
        state = np.add(state, p[l]*np.outer(outputstate, np.conj(outputstate)))

    # Compute the entropy of the state
    entropy = -1 * np.trace(np.matmul(state, linalg.logm(state)))

    # Compute the final cost
    cost = beta * np.trace(np.matmul(Hmatrix, state)) - entropy

    if (counting_num % 20 == 0) and print_info == 2:
        print('Cost at step {}: {}'.format(counting_num, cost.real))
    counting_num += 1
    return cost.real


def optimize_vqt(Hmatrix, beta, ansatz, init_params=None, shots=None,
                 method='Nelder-Mead', print_info=2):
    """
    Find the optimal parameters to produce a thermal state of the MAXCUT
    Hamiltonian for the given graph G

    Parameters
    ----------
    Hmatrix : ndarray
        Hamiltonian matrix for the MAXCUT problem
    beta : float
        Inverse temperature
    ansatz : Ansatz Object
        Instantiated ansatz object for constructing the quantum circuit
    init_params : list[float]
        Seed the variational optimization with initial values for the ansatz.
        Default is None which will select uniform random values
    shots : int
        Currently not supported, default is None which will use the
        statevector simulator
    method : str
        Scipy minimization method to use
    print_info : int
        Print intermediate values during the optimization.
        0=no info, 1=final info, 2=all info
    """

    nq = ansatz.nq
    if init_params is None:
        init_params = ansatz.gen_random_params()

    # create the optimization process
    global counting_num
    counting_num = 0
    if method in ['Nelder-Mead', 'COBYLA', 'Powell']:
        out = minimize(cost_function, x0=init_params,
                       args=(Hmatrix, beta, ansatz, print_info), method=method,
                       options={'maxiter':500})
    g = out['x']
    if print_info >= 1:
        print(out)

    return g


def prepare_learned_state(param, Hmatrix, beta, ansatz, plot=True,
                          more_info=False):
    """
    Use the optimized parameters to prepare the thermal state
    """
    nq = ansatz.nq

    angles = ansatz.parse_params(param)
    theta = angles[-1]
    phi = angles[:-1]

    opt_prob_dist = create_dist(theta)

    # Compute the probability
    p = []
    for elem in itertools.product(*opt_prob_dist):
        p.append(np.prod(elem))

    # Create empty initial density matrix
    state = np.zeros((2**nq, 2**nq))

    backend = Aer.get_backend('statevector_simulator')
    for l in range(2**nq):
        # Create the circuit
        circ = QuantumCircuit(nq)
        ansatz.construct_V(circ, [int(i) for i in list(bin(l)[2:].zfill(nq))])
        ansatz.construct_U(circ, *phi)

        # Simulate the circuit
        result = execute(circ, backend).result()
        outputstate = result.get_statevector(circ)

        # Adds a term to the "density matrix sum"
        state = np.add(state, p[l]*np.outer(outputstate, np.conj(outputstate)))

    # Compute the entropy of the state
    entropy = -1 * np.trace(np.matmul(state, linalg.logm(state)))
    ev = np.trace(np.matmul(Hmatrix, state))
    cost = beta*ev.real - entropy.real

    if plot:
        create_density_plot(state.real)
        print('Final Entropy:', entropy.real)
        print('Final Expectation Value:', ev.real)
        print('Final Cost:', cost)

    if more_info:
        return DensityMatrix(state), entropy.real, ev.real, cost
    else:
        return DensityMatrix(state)


def print_params(params, ansatz):
    ansatz.print_params(params)










