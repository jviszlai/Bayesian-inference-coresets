from active_coresets.quantum_algorithms import QuantumAlgorithm
from active_coresets.classical_algorithms import ClassicalAlgorithm
from active_coresets.quantum_algorithms import QuantumAlgorithm
from active_coresets.data_structures import Coreset, Model
from typing import List, Callable, Tuple
import numpy as np

def bayesian_inference(X: List[np.ndarray], model_from_bitstring: Callable[[str], Model], model_bitstring_len: int, m: int, A: ClassicalAlgorithm, B: QuantumAlgorithm, max_iter: int) -> Tuple[Model, Coreset]:
    coreset = Coreset()
    models = []
    most_likely_model = None
    iterations = 0
    while len(coreset.coreset) < m and iterations < max_iter:
        sampled_model, most_likely_model = B.sample_model(coreset, X, model_from_bitstring, model_bitstring_len)
        models.append(sampled_model)
        A.update_coreset(coreset, X, models)
        iterations += 1
    return most_likely_model, coreset