from __future__ import annotations
from abc import ABC, abstractmethod

import active_coresets.bayesian_coresets.bayesiancoresets as bayesiancoresets
from active_coresets.data_structures import Coreset, Distribution, Model
from active_coresets.bayesian_coresets.bayesiancoresets.coreset import HilbertCoreset
from active_coresets.bayesian_coresets.bayesiancoresets.snnls import GIGA
from active_coresets.bayesian_coresets.bayesiancoresets.projector import Projector
from typing import List, Callable
import numpy as np

class ClassicalAlgorithm(ABC):
    
    @abstractmethod
    def update_coreset(self, coreset: Coreset, X: np.ndarray, models: List[Model], k: int):
        return


class RandomSampler(ClassicalAlgorithm):

    def update_coreset(self, coreset: Coreset, X: np.ndarray, models: List[Model], k: int):
        coreset_data = coreset.unweighted_data()
        for _ in range(k):
            sample = X[np.random.choice(len(X))]
            while any((sample == x).all() for x in coreset_data):
                sample = X[np.random.choice(len(X))]
            coreset.add_data((1, sample))

class ImportanceSampler(ClassicalAlgorithm):

    def update_coreset(self, coreset: Coreset, X: np.ndarray, models: List[Distribution], k: int):
        model = models[0]
        n = X.shape[0]
        coreset_data = coreset.unweighted_data()
        samples = []
        weights = []
        for _ in range(k):
            sample = model.sample(1)[0]
            while any((sample == x).all() for x in coreset_data):
                sample = model.sample(1)
            samples.append(sample)
            weights.append(model.dist[sample])
        total_weights = np.sum(weights)
        for i in range(k):
            coreset.add_data((n * total_weights / (k * weights[i]), list(sample)))

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

    def update_coreset(self, coreset: Coreset, X: List[np.ndarray], models: List[Model], k: int):
        self.ll_projector.update_samples(models)
        new_x = []
        self.giga.build(k)
        self.giga.optimize()
        wts, pts, idcs = self.giga.get()
        
        for wt, pt, idc in zip(wts, pts, idcs):
            if idc not in self.coreset_idcs:
                self.coreset_idcs.append(idc)
                coreset.add_data((wt, pt))
                new_x.append((wt, pt))
        return new_x

    def add_unweighted_pt(self, data_idx: int):
        self.giga.snnls.w[data_idx] = 1

    def get_weight(self, data_idx: int):
        return self.giga.snnls.w[data_idx]