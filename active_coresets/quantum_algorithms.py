from abc import ABC, abstractmethod
from active_coresets.ansatz import AnsatzCircuit, ZZansatz
from active_coresets.data_structures import Coreset, Model
from typing import List, Tuple
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import minimize
import networkx as nx
import sympy
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import DensityMatrix
from active_coresets.maxcut_qaoa import solve_maxcut, gen_coreset_graph, graph_from_data

class QuantumAlgorithm(ABC):

    @abstractmethod
    def sample_model(self, coreset: Coreset, X: List[np.ndarray], Y: List[Model]) -> Tuple[Model, Model]:
        return np.random.choice(Y), np.random.choice(Y)

def generalize_model_bitstring(model_bitstring: str, coreset: Coreset, X: List[np.ndarray]) -> str:
    cluster0 = None
    cluster0_total = 0
    cluster1 = None
    cluster1_total = 0
    for i in range(len(model_bitstring)):
        if model_bitstring[-1 - i] == '1':
            if cluster1 is None:
                cluster1 = np.zeros(2)
            cluster1 += coreset.coreset[i][0] * coreset.coreset[i][1]
            cluster1_total += coreset.coreset[i][0]
        else:
            if cluster0 is None:
                cluster0 = np.zeros(2)
            cluster0 += coreset.coreset[i][0] * coreset.coreset[i][1]
            cluster0_total += coreset.coreset[i][0]
    if cluster0 is not None:
        cluster0 /= cluster0_total
    if cluster1 is not None:
        cluster1 /= cluster1_total
    generalized_model = ''
    for data in X:
        if cluster0 is None:
            generalized_model += '1'
        elif cluster1 is None:
            generalized_model += '0'
        else:
            if np.linalg.norm(data - cluster0) < np.linalg.norm(data - cluster1):
                generalized_model += '0'
            else:
                generalized_model += '1'
    print(model_bitstring)
    print(generalized_model)
    return generalized_model[::-1] # reverse endianness

class ClassicalExponentialSampler(QuantumAlgorithm):

    def plot_round(self, distribution, model_range):
        fs = 14
        fig, ax = plt.subplots(figsize=[15,5])
        plt.plot(model_range, distribution, linewidth=3, color='palegreen', label='distribution')
        plt.grid(alpha=.4, linestyle='--')
        plt.legend(fontsize=fs)
        plt.xlabel(r'Clustering model', fontsize=fs)
        plt.ylabel('Probability', fontsize=fs)
        plt.show()
        plt.close()

    def sample_model(self, coreset: Coreset, X: List[np.ndarray], Y: List[Model]) -> Tuple[Model, Model]:
        evaluations = np.array([model.evaluate_on_coreset(coreset) for model in Y])
        distribution = np.exp(evaluations)
        distribution /= np.sum(distribution)
        self.plot_round(distribution, range(len(Y)))
        return np.random.choice(Y, p=distribution), Y[np.argmax(distribution)]

class VQT(QuantumAlgorithm):

    def __init__(self, beta: int):
        self.beta = beta

    def __create_dist(self, theta):
        """
        Create the initial bitstring probability distribution parametrized by theta
        """
        prob = []
        for t in theta:
            prob.append([math.exp(-1*t), 1-math.exp(-1*t)])
        return prob

    def __cost_function(self, param, Hmatrix, beta, ansatz: AnsatzCircuit, print_info=2):
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
        opt_prob_dist = self.__create_dist(theta)

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


    def __optimize_vqt(self, Hmatrix, beta, ansatz: AnsatzCircuit, init_params=None, shots=None,
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
            out = minimize(self.__cost_function, x0=init_params,
                        args=(Hmatrix, beta, ansatz, print_info), method=method,
                        options={'maxiter':500})
        g = out['x']
        if print_info >= 1:
            print(out)

        return g

    def __create_maxcut_hamiltonian(self, H):
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


    def __gen_sympy_hamiltonian(self, n, coreset, taylor_order=0, r_plus=0.5):
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

    def __create_density_plot(self, data):
        """
        Visualize a density matrix
        """
        array = np.array(data)
        plt.matshow(array)
        plt.colorbar()
        plt.show()
        plt.close()

    def __prepare_learned_state(self, param, Hmatrix, beta, ansatz, plot=True,
                          more_info=False):
        """
        Use the optimized parameters to prepare the thermal state
        """
        nq = ansatz.nq

        angles = ansatz.parse_params(param)
        theta = angles[-1]
        phi = angles[:-1]

        opt_prob_dist = self.__create_dist(theta)

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
            self.__create_density_plot(state.real)
            print('Final Entropy:', entropy.real)
            print('Final Expectation Value:', ev.real)
            print('Final Cost:', cost)

        if more_info:
            return DensityMatrix(state), entropy.real, ev.real, cost
        else:
            return DensityMatrix(state)

    def sample_model(self, coreset: Coreset, X: List[np.ndarray], Y: List[Model]) -> Tuple[Model, Model]:
        if len(coreset.coreset) < 2:
            return super().sample_model(coreset, X, Y)
        _, G, H_str = gen_coreset_graph(coreset.coreset)
        H = self.__create_maxcut_hamiltonian(H_str)
        ansatz = ZZansatz(G)
        
        params = self.__optimize_vqt(H, self.beta, ansatz)
        density_matrix = self.__prepare_learned_state(params, H, self.beta, ansatz)
        prob_dict = density_matrix.probabilities_dict()
        best_model_bitstring = max(prob_dict, key=prob_dict.get)
        generalized_model_bitstring = generalize_model_bitstring(best_model_bitstring, coreset, X)
        generalized_model = Y[int(generalized_model_bitstring, 2)]
        return generalized_model, generalized_model



class QAOA(QuantumAlgorithm):
    def __init__(self, p: int):
        self.p = p

    def sample_model(self, coreset: Coreset, X: List[np.ndarray], Y: List[Model]) -> Tuple[Model, Model]:
        if len(coreset.coreset) < 2:
            return super().sample_model(coreset, X, Y)
        G, _ = graph_from_data(coreset.unweighted_data())
        print(G.edges.data("weight", default=1))
        print(coreset.coreset)
        best_model_bitstring = solve_maxcut(self.p, G)
        generalized_model_bitstring = generalize_model_bitstring(best_model_bitstring, coreset, X)
        generalized_model = Y[int(generalized_model_bitstring, 2)]
        return generalized_model, generalized_model