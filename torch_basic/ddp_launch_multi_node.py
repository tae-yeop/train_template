"""
참조 
https://github.com/WongKinYiu/yolov7/blob/main/train.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS
https://github.com/zzh-tech/ESTRNN/blob/master/train/ddp.py
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
- https://github.com/Jeff-sjtu/HybrIK
- https://github.com/open-mmlab/mmdetection
- https://github.com/pytorch/examples/blob/main/imagenet
- https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
- https://github.com/WongKinYiu/yolov7
- https://github.com/michuanhaohao/AICITY2021_Track2_DMT/
- https://github.com/IgorSusmelj/pytorch-styleguide
"""
"""
실행시 torch_basic까지 와서 스크립트를 실행한다고 가정
즉, working dir가 torch_basic까지 포함
"""
import sys
import os
import numpy as np
import random 
from tqdm import tqdm
# -----------------------------------------------------------------------------
# Path setting
# -----------------------------------------------------------------------------
sys.path.append(os.path.join(os.getcwd(), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda import amp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms

import argparse

from training.general import set_random_seeds, set_torch_backends, get_args, save_args_to_yaml, get_logger


def train(model, train_loader, optimizer, criterion, epoch, rank, scaler=None, sampler=None):
    """
    1-epoch training loop
    """
    model.train()
    # 새로운 텐서에도 pin memory 적용
    # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py
    ddp_loss = torch.zeros(2, device='cuda') # torch.zeros(2).to('cuda')
    if sampler:
        sampler.set_epoch(epoch)

    train_loader = tqdm(train_loader, desc = f'Epoch {epoch}', leave=False) if rank == 0 else train_loader
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to('cuda', pin_memory=True, non_blocking=True) # inputs.cuda()
        labels = labels.to('cuda', pin_memory=True, non_blocking=True) # labels.cuda()
        optimizer.zero_grad(set_to_none=True) # for param in model.parameters(): param.grad = None

        with torch.cuda.amp.autocast():
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
        # print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, training_accuracy))
        # train_loader.set_description(f'Train Epoch: {epoch} \tLoss: {training_accuracy:.6f}')
        tqdm.write(f'Train Epoch: {epoch} \tLoss: {training_accuracy:.6f}')
    return training_accuracy


def test(model, test_loader, criterion, rank):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3, device='cuda') # ddp_loss = torch.zeros(3).to('cuda')

    if rank == 0:
        pbar = tqdm(range(len(test_loader)), colour='green', desc='Validation Epoch')
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda', pin_memory=True, non_blocking=True)
            labels = labels.to('cuda', pin_memory=True, non_blocking=True)
            output = model(inputs)
            ddp_loss[0] += criterion(output, labels).item() * len(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(labels.view_as(pred)).sum().item()
            ddp_loss[2] += len(inputs)

            if rank == 0:
                pbar.update(1)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    test_loss = ddp_loss[0] / ddp_loss[2]
    if rank == 0:
        pbar.close()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

    return test_loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    args = get_args()

    # ============================ Distributed Setting ============================
    # 디폴트 init_method 'env://' 사용
    dist.init_process_group(backend='nccl')
    global_rank = int(os.environ['RANK']) # dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK']) # device_id = global_rank % torch.cuda.device_count()
    world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)


    # ============================ Basic Setting ==================================
    seed = args.random_seed
    if seed is not None:
        set_random_seeds(seed) 

    set_torch_backends(args)

    if args.wandb and global_rank == 0:
        wandb.init(project=)

    
    # ============================ Dataset =========================================

    assert args.batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    
    if dist.get_rank() != 0:
        dist.barrier()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)

    if dist.get_rank() == 0:
        dist.barrier()

    batch_size = 512
    # DistributedSampler의 디폴트 shuffle는 True임
    train_sampler = DistributedSampler(trainset, rank=global_rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(testset, rank=global_rank, num_replicas=world_size, shuffle=False)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, 
                              num_workers=4, pin_memory=True, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, 
                             num_workers=4, pin_memory=True, shuffle=False)



    # ============================ Models ============================================
    model = Net().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    # model = DDP(model, device_ids=[local_rank], 
    #             output_device=local_rank,
    #             find_unused_parameters=isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
    
    # ============================ Traning setup ======================================

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # ============================ Resume setup ========================================
    ckpt_dir = None
    ckpt_filename = None
    resume = False
    if ckpt_dir is not None and ckpt_filename is not None:
        ckpt_filepath = os.path.join(ckpt_dir, ckpt_filename)
    if resume == True:
        # 체크포인트에서 텐서와 메타데이터 가져옴
        # 텐서안에 storage라는 오브젝트가 있는데 이것의 위치를 장치로 옮겨줌
        map_location = lambda storage, loc: storage.cuda(local_rank)
        checkpoint = torch.load(ckpt_filepath, map_location=map_location) # = torch.load(ckpt_filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    # ============================ Train ==============================================
    for epoch in range(10):
        train(model, train_loader, optimizer, loss, epoch, global_rank, train_sampler)
        test()
        scheduler.step()

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.module.state_dict(), # 이렇게 해야 나중에 싱글 추론 가능
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict' : scheduler.state_dict()
                      }
        
    dist.barrier()
    init_end_event.record()

    if global_rank==0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    dist.destroy_process_group()