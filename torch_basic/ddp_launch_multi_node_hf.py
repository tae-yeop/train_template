"""
참조 
https://github.com/WongKinYiu/yolov7/blob/main/train.py
https://github.com/CUBOX-Co-Ltd/cifar10_DDP_FSDP_DS
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda import amp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset


# Ampare architecture 30xx, a100, h100,..
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# infernce
# torch.set_grad_enabled(False)


def train(tokenizer, model, train_loader, optimizer, criterion, epoch, rank, scaler=None, sampler=None):
    model.train()
    if sampler:
        sampler.set_epoch(epoch)
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        tokens = tokenizer(
            data["premise"],
            data["hypothesis"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",)

        loss = model(input_ids=tokens.input_ids.cuda(),
                     attention_mask=tokens.attention_mask.cuda(),
                     labels=data["labels"],).loss

        if scaler:
            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if idx % 10 == 0 and rank == 0:
            print(f"step:{idx}, loss:{loss}")

# 디폴트 init_method 'env://' 사용
dist.init_process_group(backend='nccl')
global_rank = int(os.environ['RANK']) # dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)


# Dataset

datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]


batch_size = 512
train_sampler = DistributedSampler(datasets, num_replicas=world_size)
# test_sampler = DistributedSampler(testset, shuffle=False)
train_loader = DataLoader(datasets, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
# test_loader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)


model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='./tmp')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3, cache_dir='./tmp').cuda()
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
# model = DDP(model, device_ids=[local_rank], 
#             output_device=local_rank,
#             find_unused_parameters=isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
# nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698


loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


init_start_event = torch.cuda.Event(enable_timing=True)
init_end_event = torch.cuda.Event(enable_timing=True)
init_start_event.record()

for epoch in range(10):
    train(tokenizer, model, train_loader, optimizer, epoch, global_rank, train_sampler)
    scheduler.step()

dist.barrier()
init_end_event.record()

if global_rank==0:
    torch.save(model.state_dict(), 'model.pt')
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
dist.destroy_process_group()
