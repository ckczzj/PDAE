from abc import ABC, abstractmethod

import torch

from utils import set_seed

class BaseSampler(ABC):
    def __init__(self, config, distributed_meta_info):
        super().__init__()

        self.config = config
        self.distributed_meta_info = distributed_meta_info

        self.rank = self.distributed_meta_info["rank"]
        assert self.rank == torch.distributed.get_rank()
        self.world_size = self.distributed_meta_info["world_size"]
        assert self.world_size == torch.distributed.get_world_size()

        self.device = torch.device('cuda:{}'.format(self.rank))
        torch.cuda.set_device(self.device)

        self._build_everything()

        # after building model, set different seed for different ranks for training or sampling with different noise
        set_seed(self.rank)
        torch.distributed.barrier()

    def _build_everything(self):
        self._build_dataloader()
        print('rank{}: dataloader built.'.format(self.rank))
        self._build_model()
        print('rank{}: model built.'.format(self.rank))

    @abstractmethod
    def _build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @staticmethod
    def dispatch_num_samples_for_process(num_samples, num_process, rank):
        assert 1 <= num_process <= num_samples
        assert 0 <= rank <= num_process - 1
        average = num_samples // num_process
        remainder = num_samples % num_process
        dispatch = [average] * num_process
        if remainder > 0:
            dispatch[-1] += remainder
        assert sum(dispatch) == num_samples
        return dispatch[rank]

    # return a list of data from all ranks
    def gather_data(self, data):
        gather_data = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gather_data, data)
        return gather_data
