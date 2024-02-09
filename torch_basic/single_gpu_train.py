# 참조 : https://github.com/HazyResearch/state-spaces/blob/main/example.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint

import torchvision
import torchvision.transforms as transforms

import os
import argparse

'''
Arguments
'''
parser = argparse.ArgumentParser(decription)

parser.add_argument('path', type=str, help)
parser.add_argument('--phase', type=str, help)
parser.add_argument('--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'] )
parser.add_argument('--amp', type=str, default='fp16', choices=['fp16'])
parser.add_argument('--act_ckpt', type=str, default=False)
args = parser.parse_args()

# 
torch.backends.cuda.matmul.allow_tf32 = True

'''
reproducibility
- https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31 참조
'''
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
'''
Model
'''
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_checkpoint = args.act_ckpt

        self.layers = nn.ModuleList([])
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
                

class Head(nn.Module):
    def 


# Init the model
g_running = StyledGenerator(code_size).cuda()
g_running.train(False)


'''
Optimizer
'''
g_optimizer = optim.Adam(generator.module.gerator.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

g_optimizer.add_param_group({
    'params'
})

# Learning rate scheduling
## from scratch
def adjust_lr(optimizer, lr):
  for group in opimizer
    mult = group.get('mult', 1)
    group['lr'] = lr * mult

## optim.lr_scheduler


# Option에 따른 다른 loss 사용
if args.loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss()

'''
dataset & dataloader
'''

# Custom Dataset
trainset = Dataset(...)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2,
                                              pin_memory=True) 


# MNIST Dataset
trasform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False)


"""
train one epoch
- mixed precision
"""
def train(args, model, device, train_loader, optimizer, epoch):

    # 이 부분을 여기서 해야하나? 밖에서 한번만 하면?
    if args.amp == 'fp16':
        amp_enabled = True
        # amp.GradScale : underfitting을 방지하는 용도
        scaler = amp.GradScaler(amp_enabled) # or amp.grad_scaler.GradScaler(amp_enabled)
        
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data)
            # loss를 autocast 안에 넣어야?
            # 특정 레이어의 경우 float32로 계산하기 때문에 밖에 둬도 상관없는듯
            # 그런데 기본은 안에 두는 것인듯 
            loss = ...

        if args.amp == 'fp16':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # gradient norm 적용을 위해서 https://pytorch.org/docs/stable/amp.html
        else:
            loss.backward()
            
        # gradient norm
        if args.gradient_norm == True:

            
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0:
            print(
                'Train epoch: {} [{}/{} (:.0f)%]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
                )
            )
        
def test(model, device, test_loader):




'''
training loop
- epoch단위로 할건지
'''
pbar = tqdm(range(start_epoch, ))

from tqdm.auto import tqdm
progress_bar = tqdm(
    range(global_step, args.max_train_steps),
    initial=initial_global_step,
    disable=not accelerator.is_local_main_process
)
progress_bar.set_description('steps')

for epoch in range(start_epoch, epochs):
    train(args, model)
    test(model, test_loader)
    scheduler.step()