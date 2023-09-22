import numpy as np
import lpips

from metric.base_metric import BaseMetric
from metric.utils import calculate_lpips

class LPIPSMetric(BaseMetric):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

    def process(self, targets, samples):
        lpips = calculate_lpips(targets, samples, self.lpips_fn)
        self.results.extend(lpips.tolist())

    def compute_metrics(self, results):
        return np.mean(results)