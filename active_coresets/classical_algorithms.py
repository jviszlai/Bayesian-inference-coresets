from abc import ABC, abstractmethod
from active_coresets.data_structures import Coreset, Model
from typing import List
import numpy as np

class ClassicalAlgorithm(ABC):
    
    @abstractmethod
    def update_coreset(self, coreset: Coreset, X: List[np.ndarray], models: List[Model]):
        return


class RandomSampler(ClassicalAlgorithm):

    def update_coreset(self, coreset: Coreset, X: List[np.ndarray], models: List[Model]):
        coreset_data = coreset.unweighted_data()
        sample = X[np.random.choice(len(X))]
        while any((sample == x).all() for x in coreset_data):
            sample = X[np.random.choice(len(X))]
        coreset.add_data((1, sample))
