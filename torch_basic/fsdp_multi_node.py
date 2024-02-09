"""
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS/blob/master/fsdp_cifar10.py
https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_tutorial.rst
https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_adavnced_tutorial.rst
https://github.com/pytorch/examples/blob/main/distributed/FSDP/utils/train_utils.py
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
    enable_wrap,
    wrap,
    ModuleWrapPolicy
)

import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from functools import partial

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def train(model, train_loader, optimizer, criterion, epoch, rank, scaler=None, sampler=None):
    model.train()
    # ddp_loss = torch.zeros(2).to('cuda')
    ddp_loss = torch.zeros(2, device='cuda')
    if sampler:
        sampler.set_epoch(epoch)
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda', pin_memory=True, non_blocking=True)
        labels = labels.to('cuda', pin_memory=True, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        output = model(inputs)
        loss = criterion(output, labels)
        if scaler:
            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(inputs)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    training_accuracy = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, training_accuracy))
    return training_accuracy

def test(model, test_loader, criterion, rank):
    model.eval()
    correct = 0
    # ddp_loss = torch.zeros(3).to('cuda')
    ddp_loss = torch.zeros(3, deviec='cuda')
    with torch.no_grad():
        for data, target in test_loader:
            inputs = inputs.to('cuda', pin_memory=True, non_blocking=True)
            labels = labels.to('cuda', pin_memory=True, non_blocking=True)
            output = model(data)
            ddp_loss[0] += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

    return test_loss

global_rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if global_rank == 0:
    torchvision.datasets.CIFAR10(root='./data',train=True,download=True,)
    torchvision.datasets.CIFAR10(root='./data',train=False,download=True,)
dist.barrier()

batch_size = 512
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=False, transform=transform)
train_sampler = DistributedSampler(trainset)
test_sampler = DistributedSampler(testset, shuffle=False)
train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

# https://pytorch.org/vision/stable/_modules/torchvision/models/swin_transformer.html#swin_v2_b
model = tvm.swin_v2_b()
model.head = nn.Linear(1024, 10)

from torchvision.models.swin_transformer import SwinTransformerBlock
mp_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={SwinTransformerBlock,})

fsdp_config = {
        "cpu_offload": CPUOffload(offload_params=True),  # Uncomment for CPU offloading
        "mixed_precision": MixedPrecision(param_dtype=torch.bfloat16,
                                          reduce_dtype=torch.bfloat16,
                                          buffer_dtype=torch.bfloat16),
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,  # Specify if different from FULL_SHARD
        "auto_wrap_policy": t5_auto_wrap_policy,
        "device_id": torch.cuda.current_device(),
    }

model = FSDP(model, **fsdp_config)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

scaler = ShardedGradScaler(init_scale=2.0)
best_val_loss = float("inf")
curr_val_loss = float("inf")
save_model = False
file_save_name = "SwinT2-"
time_of_run = get_date_of_run()

init_start_event = torch.cuda.Event(enable_timing=True)
init_end_event = torch.cuda.Event(enable_timing=True)

init_start_event.record()
for epoch in range(epochs):
    train_accuracy = train(model, train_loader, optimizer, loss, epoch, global_rank, train_sampler)
    curr_val_loss = test(model, test_loader, loss, global_rank)
    scheduler.step()
        
    if save_model and curr_val_loss < best_val_loss:
        dist.barrier()
        states = model.state_dict()
        if global_rank==0:
            torch.save(states, "model.pt")

        if global_rank == 0:
            print(f"--> entering save model state")

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
            cpu_state = model.state_dict()

        if global_rank == 0:
            print(f"--> saving model ...")
            currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
            print(f"--> attempting to save model prefix {currEpoch}")
            save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
            print(f"--> saving as model name {save_name}")

            torch.save(cpu_state, save_name)

    if curr_val_loss < best_val_loss:
        best_val_loss = curr_val_loss
        if global_rank==0:
            print(f"-->>>> New Val Loss Record: {best_val_loss}")


dist.barrier()
init_end_event.record()

if global_rank==0:
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
dist.destroy_process_group()
