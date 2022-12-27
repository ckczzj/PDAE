import sys
sys.path.append("../")

import argparse
import os

import torch
import torch.multiprocessing as mp

from representation_learning_trainer import RepresentationLearningTrainer
from utils import load_yaml, init_process

def run(rank, args, distributed_meta_info):
    distributed_meta_info["rank"] = rank
    init_process(
        init_method=distributed_meta_info["init_method"],
        rank=distributed_meta_info["rank"],
        world_size=distributed_meta_info["world_size"]
    )

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


    restore = os.path.exists(os.path.join(os.path.join(args.run_path, 'checkpoints'), 'latest.pt'))
    config = load_yaml(os.path.join(args.run_path, "config.yml")) if restore else load_yaml(args.config_path)
    runner = RepresentationLearningTrainer(
        config=config,
        run_path=args.run_path,
        distributed_meta_info=distributed_meta_info
    )

    runner.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=os.path.join("../config", "ffhq_representation_learning.yml"))
    # Automatically detect run_path/checkpoints/latest.pt to decide whether
    # to restore or
    # to train from scratch using config.yml above.
    parser.add_argument('--run_path', type=str, default=os.path.join("../runs", "ffhq_representation_learning"))
    parser.add_argument('--world_size', type=str, required=True)
    parser.add_argument('--master_addr', type=str, default="127.0.0.1")
    parser.add_argument('--master_port', type=str, default="6666")

    args = parser.parse_args()

    world_size = int(args.world_size)
    init_method = "tcp://{}:{}".format(args.master_addr, args.master_port)

    distributed_meta_info = {
        "world_size": world_size,
        "master_addr": args.master_addr,
        "init_method": init_method,
        # rank will be added in spawned processes
    }

    mp.spawn(
        fn=run,
        args=(args, distributed_meta_info),
        nprocs=world_size,
        join=True,
        daemon=False
    )

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_representation_learning.py --world_size 4
