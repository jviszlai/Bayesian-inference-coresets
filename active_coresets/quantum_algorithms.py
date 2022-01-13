from abc import ABC, abstractmethod
from active_coresets.ansatz import AnsatzCircuit, ZZansatz
from active_coresets.data_structures import Coreset, Model
from typing import Callable, List, Tuple
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
from active_coresets.maxcut_qaoa import solve_maxcut, gen_coreset_graph, graph_from_weighted_data
from active_coresets.maxcut_vqt import create_maxcut_hamiltonian, optimize_vqt, prepare_learned_state
from active_coresets.QBM import QuantumBoltzmannMachine

class QuantumAlgorithm(ABC):

    @abstractmethod
    def sample_model(self, coreset: Coreset, X: List[np.ndarray], model_from_bitstring: Callable[[str], Model], model_bitstring_len: int) -> Tuple[Model, Model]:
        bitstring = ''
        for i in range(model_bitstring_len):
            if np.random.rand() > 0.5:
                bitstring += '1'
            else:
                bitstring += '0'
        model = model_from_bitstring(bitstring)
        return model, model

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

    def sample_model(self, coreset: Coreset, X: List[np.ndarray], model_from_bitstring: Callable[[str], Model], model_bitstring_len: int) -> Tuple[Model, Model]:
        if len(coreset.coreset) < 2:
            return super().sample_model(coreset, X, model_from_bitstring, model_bitstring_len)
        _, G, H_str = gen_coreset_graph(coreset.coreset)
        H = create_maxcut_hamiltonian(H_str)

        ansatz = ZZansatz(G)
        
        params = optimize_vqt(H, self.beta, ansatz)
        density_matrix = prepare_learned_state(params, H, self.beta, ansatz)
        prob_dict = density_matrix.probabilities_dict()
        best_model_bitstring = max(prob_dict, key=prob_dict.get)
        generalized_model_bitstring = generalize_model_bitstring(best_model_bitstring, coreset, X)
        generalized_model = model_from_bitstring(generalized_model_bitstring)
        return generalized_model, generalized_model


class QAOA(QuantumAlgorithm):
    def __init__(self, p: int):
        self.p = p

    def sample_model(self, coreset: Coreset, X: List[np.ndarray], model_from_bitstring: Callable[[str], Model], model_bitstring_len: int) -> Tuple[Model, Model]:
        if len(coreset.coreset) < 2:
            return super().sample_model(coreset, X, model_from_bitstring, model_bitstring_len)
        weights = [weight for weight, data in coreset.coreset]
        data = [data for weight, data in coreset.coreset]
        G, _ = graph_from_weighted_data(data, weights)
        #TODO: fix weighting for future coresets with non-uniform weights
        print(G.edges.data("weight", default=1))
        print(coreset.coreset)
        best_model_bitstring = solve_maxcut(self.p, G)
        generalized_model_bitstring = generalize_model_bitstring(best_model_bitstring, coreset, X)
        generalized_model = model_from_bitstring(generalized_model_bitstring)
        return generalized_model, generalized_model

class RQBM(QuantumAlgorithm):
    def __init__(self, graph: nx.Graph, visible_nodes: List[int], hidden_nodes: List[int], beta: float = 10.0):
        self.beta = beta
        #self.rqbm = QuantumBoltzmannMachine(graph, visible_nodes, hidden_nodes)
        self.graph = graph
        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes


    def sample_model(self, coreset: Coreset, X: List[np.ndarray], model_from_bitstring: Callable[[str], Model], model_bitstring_len: int) -> Tuple[Model, Model]:
        self.rqbm = QuantumBoltzmannMachine(self.graph, self.visible_nodes, self.hidden_nodes)
        self.rqbm.exact_train_coreset(coreset=coreset, beta=self.beta, verbose=0)
        model_dist = self.rqbm.get_distribution(self.beta)
        
        # For debugging
        data_dict = {}
        for weight, pt in coreset.coreset:
            z_pt = tuple(-2 * pt + np.ones(pt.shape[0]))
            if z_pt in data_dict:
                data_dict[z_pt] += weight
            else:
                data_dict[z_pt] = weight
        data_dist = {k: v / sum(data_dict.values()) for k, v in data_dict.items()}

        print(f"Model dist: {model_dist}")
        print(f"Data dist: {data_dist}")
        print(f"Coreset: {coreset.coreset}")
        self.rqbm.plot_dist(model_dist, data_dist)
        # Ugly workaround for now. TODO: not this
        model = model_from_bitstring(self.rqbm, self.beta)
        return model, model