import os
import random
import numpy as np

import torch
import torch.distributed as dist


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_distributed_mode(dist_cfg):
    """Initialize torch.distributed. Expects the process to be launched with
    torchrun, which sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT.
    Falls back to single-process mode when run as plain `python main.py`.
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        init_method = 'env://'
    else:
        rank, world_size, local_rank = 0, 1, 0
        init_method = 'tcp://127.0.0.1:29500'

    dist_cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=dist_cfg.backend, init_method=init_method, world_size=world_size, rank=rank)

    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
