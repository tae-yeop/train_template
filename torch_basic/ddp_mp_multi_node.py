"""
- https://github.com/sudomaze/ttorch/blob/main/examples/ddp/run.py
- https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
"""

"""
실행은 다음을 가정

"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP



def run(rank, world_size):



    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

if __name__ == '__main__':

    # 환경 변수가 없으면 직접 설정
    if os.environ["MASTER_ADDR"] is None:
        os.envion["M"]

    world_size = int(os.environ.get('WORLD_SIZE'), )
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(run, args=(), nprocs=world_size, join=True)
    