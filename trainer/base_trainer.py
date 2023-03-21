import os
import time
import copy
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataset as dataset_module
from utils import save_yaml, set_seed, set_worker_seed_builder


class BaseTrainer(ABC):
    def __init__(self, config, run_path, distributed_meta_info):
        super().__init__()

        # set the same seed for different ranks for training with the same parameter initialization
        set_seed(0)

        self.config = config
        self.run_path = run_path
        self.distributed_meta_info = distributed_meta_info

        self.rank = self.distributed_meta_info["rank"]
        assert self.rank == torch.distributed.get_rank()
        self.world_size = self.distributed_meta_info["world_size"]
        assert self.world_size == torch.distributed.get_world_size()

        self.device = torch.device('cuda:{}'.format(self.rank))
        torch.cuda.set_device(self.device)

        self.step = 0
        self.writer = None

        # if self.is_sampling: # sampling
        #     self._build_everything()
        #     self.load(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt'))
        if os.path.exists(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt')): # restore to train
            self._build_everything()
            self.load(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt'))
            if self.rank == 0:
                self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, 'tb'), flush_secs=10, purge_step=self.step + 1)
        else: # train from scratch
            if self.rank == 0:
                os.makedirs(os.path.join(self.run_path, 'checkpoints'), exist_ok=True)
                os.makedirs(os.path.join(self.run_path, 'samples'), exist_ok=True)
                save_yaml(os.path.join(self.run_path, 'config.yml'), self.config)
                self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, 'tb'), flush_secs=10)
                self.writer.add_text('config', str(self.config))
                self.writer.flush()
            self._build_everything()

        # after building model, set different seed for different ranks for training or sampling with different noise
        set_seed(self.rank)
        torch.distributed.barrier()

    def _build_everything(self):
        self._build_dataloader()
        print('rank{}: dataloader built.'.format(self.rank))
        self._build_model()
        print('rank{}: model built.'.format(self.rank))
        self._build_optimizer()
        print('rank{}: optimizer built.'.format(self.rank))

    def _build_dataloader(self):
        train_dataset_config = self.config["train_dataset_config"]
        eval_dataset_config = copy.deepcopy(train_dataset_config)
        eval_dataset_config.update(self.config["eval_dataset_config"])
        dataset_name = train_dataset_config["name"]

        train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
        eval_dataset = getattr(dataset_module, dataset_name, None)(eval_dataset_config)

        # dispatch batch_size
        dataloader_config = copy.deepcopy(self.config["dataloader_config"])
        global_batch_size = dataloader_config["batch_size"]
        local_batch_size = global_batch_size // self.world_size
        if global_batch_size % self.world_size != 0:
            local_batch_size += 1
        assert local_batch_size > 0
        dataloader_config["batch_size"] = local_batch_size

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=self.train_sampler,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            worker_init_fn=set_worker_seed_builder(self.rank),
            persistent_workers=True,
            **dataloader_config
        )
        self.train_dataloader_infinite_cycle = self.build_train_dataloader_infinite_cycle()

        self.eval_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=eval_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )

        self.eval_dataloader = DataLoader(
            dataset=eval_dataset,
            sampler=self.eval_sampler,
            pin_memory=False,
            collate_fn=eval_dataset.collate_fn,
            # main process is enough for sampling, no subprocesses spawn, no need for worker_init_fn, no need for persistent_workers
            num_workers=0,
            # 36: showing a 6x6 image grid in tensorboard
            batch_size=self.dispatch_num_samples_for_process(36, self.world_size, rank=self.rank)
        )

    def build_train_dataloader_infinite_cycle(self):
        if self.rank == 0:
            base_seed = [int(time.time())]
        else:
            base_seed = [None]
        torch.distributed.broadcast_object_list(base_seed, src=0, device=self.device)

        base_seed = base_seed[0]
        print(self.rank, "data_loader_seed", base_seed)
        while True:
            base_seed += 1
            self.train_sampler.set_epoch(base_seed)
            for data in self.train_dataloader:
                yield data

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    @abstractmethod
    def _build_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
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
