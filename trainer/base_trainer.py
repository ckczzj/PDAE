import os
import time
import copy
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

import dataset as dataset_module
from utils.utils import load_yaml, save_yaml, set_seed, set_worker_seed_builder, init_distributed_mode


class BaseTrainer(ABC):
    def __init__(self, args):
        super().__init__()
        init_distributed_mode(args)

        self.global_rank = args.global_rank
        self.global_world_size = args.global_world_size
        self.local_rank = args.local_rank
        self.device = torch.device('cuda:{}'.format(self.local_rank))
        torch.cuda.set_device(self.device)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # set the same seed for different ranks for training with the same parameter initialization
        set_seed(0)

        self.config = load_yaml(args.config_path)
        self.run_path = args.run_path
        self.step = 0
        self.writer = None

        if args.resume:  # restore to train
            self._build_everything()
            self.load(args.resume)
            if self.global_rank == 0:
                self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, 'tb'), flush_secs=10, purge_step=self.step + 1)
        else:  # train from scratch
            if self.global_rank == 0:
                os.makedirs(os.path.join(self.run_path, 'checkpoints'), exist_ok=True)
                os.makedirs(os.path.join(self.run_path, 'samples'), exist_ok=True)
                save_yaml(os.path.join(self.run_path, 'config.yml'), self.config)
                self.writer = SummaryWriter(log_dir=os.path.join(self.run_path, 'tb'), flush_secs=10)
                self.writer.add_text('config', str(self.config))
                self.writer.flush()
            self._build_everything()

        # after building model, set different seed for different ranks for training or sampling with different noise
        set_seed(self.global_rank)
        torch.distributed.barrier()

    def _build_everything(self):
        self._build_dataloader()
        print('rank{}: dataloader built.'.format(self.global_rank))
        self._build_model()
        print('rank{}: model built.'.format(self.global_rank))
        self._build_optimizer()
        print('rank{}: optimizer built.'.format(self.global_rank))

    def _build_dataloader(self):
        train_dataset_config = self.config["train_dataset_config"]
        eval_dataset_config = copy.deepcopy(train_dataset_config)
        eval_dataset_config.update(self.config["eval_dataset_config"])
        dataset_name = train_dataset_config["name"]

        self.train_dataset = getattr(dataset_module, dataset_name, None)(train_dataset_config)
        self.eval_dataset = getattr(dataset_module, dataset_name, None)(eval_dataset_config)

        self.global_batch_size = self.config["dataloader_config"]["train"]["batch_size"] * self.global_world_size

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.global_world_size,
            rank=self.global_rank,
            shuffle=True,
            drop_last=True,  # DistributedSampler drop_last means whether to drop the last len(dataset) % num_replicas samples
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
            worker_init_fn=set_worker_seed_builder(self.global_rank),
            persistent_workers=True,
            drop_last=True,  # Dadaloader drop_last means whether to drop the last non-full batch
            **self.config["dataloader_config"]["train"]
        )
        self.train_dataloader_infinite_cycle = self.build_train_dataloader_infinite_cycle()

        self.eval_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=self.eval_dataset,
            num_replicas=self.global_world_size,
            rank=self.global_rank,
            shuffle=True,
            drop_last=True,
        )

        self.eval_dataloader = torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            sampler=self.eval_sampler,
            pin_memory=False,
            collate_fn=self.eval_dataset.collate_fn,
            # main process is enough for sampling, no subprocesses spawn, no need for worker_init_fn, no need for persistent_workers
            num_workers=0,
            batch_size=self.dispatch_num_samples_for_process(self.config["dataloader_config"]["eval"]["num_generations"], self.global_world_size, rank=self.global_rank),
            drop_last=True,
        )

    def build_train_dataloader_infinite_cycle(self):
        if self.global_rank == 0:
            base_seed = [int(time.time())]
        else:
            base_seed = [None]
        torch.distributed.broadcast_object_list(base_seed, src=0, device=self.device)

        base_seed = base_seed[0]
        print(self.global_rank, "data_loader_seed", base_seed)
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
        gather_data = [None for _ in range(self.global_world_size)]
        torch.distributed.all_gather_object(gather_data, data)
        return gather_data
