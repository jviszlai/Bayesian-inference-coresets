import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

class Coreset:
    
    def __init__(self, initial_data: List[Tuple[float, np.ndarray]] = None):
        """
        Base class for a coreset - a weighted subset of some data 

        Parameters
        ----------
        initial_data: List[Tuple[float, np.ndarray]]
            Initial data comprised of (data_pt, weight) tuples
        
        """
        if not initial_data:
            self.coreset = []
        else:
            self.coreset = initial_data
    
    def unweighted_data(self) -> List[np.ndarray]:
        """
        Returns an array of the unweighted data points in the coreset
        """
        return [weighted_pt[1] for weighted_pt in self.coreset]
    
    def add_data(self, weighted_point: Tuple[float, np.ndarray]):
        """
        Adds weighted_point to the coreset
        
        Parameters
        ----------
        weighted_point: a (weight, data_pt) tuple 
        """
        self.coreset.append(weighted_point)

class Model(ABC):
    
    @abstractmethod
    def __init__(self, model):
        """
        Abstract class for user-defined model
        """
        self.model = model
    
    @abstractmethod
    def evaluate_on_coreset(self, coreset: Coreset) -> float:
        return 0.0

