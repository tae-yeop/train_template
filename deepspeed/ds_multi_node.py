"""
https://github.com/microsoft/DeepSpeedExamples/blob/master/cifar/cifar10_deepspeed.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS/blob/master/deepspeed_cifar10.py
https://github.com/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/08_zero_redundancy_optimization.ipynb
"""
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

import deepspeed



import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms

import datetime

def get_ds_config(args):
    ds_config = {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2000,
        "optimizer":{
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler":{
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr":0,
                "warmup_max_lr":0.001,
                "warmup_num_steps":1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"}, # args에 따라서 True 걸리도록
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
        "activation_checkpointing":{
            "partition_activations": True,
            "cpu_checkpointing": True, # 매우 큰 activation 텐서는 CPU로 오프로딩함
            
        },
        
    }
    return ds_config
    

def get_date_of_run():
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def train(model, train_loader, rank):
    if rank == 0:
        train_loader = tqdm(train_loader)

    ddp_loss = torch.zeros(2).to('cuda')
    

# 이게 반드시 필요할까?
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


deepspeed.init_distributed(dist_backend='nccl')
local_rank = args.local_rank
torch.cuda.set_device(local_rank)


transform = transforms.Compose([
    transforms.ToTensor(),
    trnasforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if dist.get_rank() != 0:
    dist.barrier()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)

if dist.get_rank() == 0:
    dist.barrier()


train_sampler = DistributedSampler(trainset)
test_sampler = DistributedSampler(testset, shuffle=False)
train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=4)


class Net(nn.Module):
    def __init__(self, config):
        
model = tvm.swin_v2_b()
model.head = nn.Linear(1024, 10)

parameters = model.parameters()
model_engine, optimizer, trainloader, scheduler = deepspeed.initialize(
    args=args,
    model=net, 
    model_parameters=parameters,
    training_data = trainset
)

# dz_fp16 = model_engine.fp16_enabled()

criterion = nn.CrossEntropyLoss()
