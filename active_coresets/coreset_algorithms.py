from active_coresets.quantum_algorithms import QuantumAlgorithm
from active_coresets.classical_algorithms import ClassicalAlgorithm
from active_coresets.quantum_algorithms import QuantumAlgorithm
from active_coresets.data_structures import Coreset, Model
from typing import List
import numpy as np

def bayesian_inference(X: List[np.ndarray], Y: List[Model], m: int, A: ClassicalAlgorithm, B: QuantumAlgorithm) -> Model:
    coreset = Coreset()
    models = []
    most_likely_model = None
    for _ in range(m):
        sampled_model, most_likely_model = B.sample_model(coreset, X, Y)
        models.append(sampled_model)
        A.update_coreset(coreset, X, models)
    return most_likely_model