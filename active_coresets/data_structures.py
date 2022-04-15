import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

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
    
    def evaluate_on_coreset(self, coreset: Coreset) -> float:
        return 0.0

    def evaluate_on_point(self, data: np.ndarray) -> float:
        return 0.0

class Distribution(Model):
    def __init__(self, dist: Dict[int, float]):
        self.dist = dist
    
    def sample(self, num_samples: int) -> np.ndarray:
        """
        Samples from the distribution
        
        Parameters
        ----------
        num_samples: int
            Number of samples to sample
        
        Returns
        -------
        samples: np.ndarray
            Samples from the distribution
        """
        keys = list(self.dist.keys())
        sample_ids = np.random.choice(range(len(keys)), num_samples, p=list(self.dist.values()))
        return [keys[id] for id in sample_ids]