from __future__ import annotations
from abc import ABC, abstractmethod

import active_coresets.bayesian_coresets.bayesiancoresets as bayesiancoresets
from active_coresets.data_structures import Coreset, Model
from bayesiancoresets.coreset import HilbertCoreset
from bayesiancoresets.snnls import GIGA
from bayesiancoresets.projector import Projector
from typing import List, Callable
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

class GIGACoreset(ClassicalAlgorithm):

    class QuantumSampleProjector(Projector):

        def __init__(self, loglikelihood: Callable[[np.ndarray, List[Model]], np.ndarray]):
            self.loglikelihood = loglikelihood

        def project(self, pts: np.ndarray) -> np.ndarray:
            lls = self.loglikelihood(pts, self.samples)
            if len(self.samples) > 1:
                lls -= lls.mean(axis=1)[:,np.newaxis]
            return lls

        # Since sampling is done by quantum algo this method does job of the update method
        def update_samples(self, models: List[Model]):
            self.samples = models

    class QuantumHilbertCoreset(HilbertCoreset):

        def __init__(self, data: np.ndarray, ll_projector: GIGACoreset.QuantumSampleProjector, snnls_algo=GIGA, **kw):
            self.data = data
            self.ll_projector = ll_projector
            self.snnls_algo = snnls_algo
            self.snnls = None
            bayesiancoresets.coreset.coreset.Coreset.__init__(self, **kw)
        
        def _build(self, itrs: int):
            sub_idcs = np.arange(self.data.shape[0])
            vecs = self.ll_projector.project(self.data)
            if self.snnls:
                self.w = self.snnls.weights()
            else:
                self.w = np.zeros(self.data.shape[0])
            self.snnls = self.snnls_algo(vecs.T, vecs.sum(axis=0))
            self.snnls.w = self.w
            self.sub_idcs = sub_idcs
            super()._build(itrs)


    def __init__(self, X: List[np.ndarray], loglikelihood: Callable[[np.ndarray, List[Model]], np.ndarray]):
        self.ll_projector = GIGACoreset.QuantumSampleProjector(loglikelihood)
        self.giga = GIGACoreset.QuantumHilbertCoreset(np.array(X), self.ll_projector)
        self.coreset_idcs = []

    def update_coreset(self, coreset: Coreset, X: List[np.ndarray], models: List[Model]):
        self.ll_projector.update_samples(models)
        self.giga.build(1)
        wts, pts, idcs = self.giga.get()
        for wt, pt, idc in zip(wts, pts, idcs):
            if idc not in self.coreset_idcs:
                self.coreset_idcs.append(idc)
                coreset.add_data((wt, pt))