from abc import ABC, abstractmethod

import torch

class BaseMetric(ABC):
    def __init__(self):
        super().__init__()
        self.results = []

    def all_gather_results(self, world_size):
        gather_results_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_results_list, self.results)
        gather_results = []
        for results in gather_results_list:
            gather_results.extend(results)
        return gather_results

    @abstractmethod
    def process(self):
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self):
        raise NotImplementedError

    def reset(self):
        self.results = []
