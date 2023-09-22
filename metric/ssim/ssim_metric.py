import numpy as np

from metric.base_metric import BaseMetric
from metric.utils import calculate_ssim

class SSIMMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, targets, samples):
        ssim = calculate_ssim(targets, samples)
        self.results.extend(ssim.tolist())

    def compute_metrics(self, results):
        return np.mean(results)