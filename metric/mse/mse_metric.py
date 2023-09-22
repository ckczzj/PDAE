import numpy as np

from metric.base_metric import BaseMetric
from metric.utils import calculate_mse

class MSEMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, targets, samples):
        mse = calculate_mse(targets, samples)
        self.results.extend(mse.tolist())

    def compute_metrics(self, results):
        return np.mean(results)