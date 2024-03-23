"""
참조
https://github.com/HazyResearch/state-spaces/blob/main/example.py
https://github.com/WongKinYiu/yolov7/blob/main/train.py
https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/eb9cafbb61dfb9722afb5fb21eff75ed999ad52f/src/loader.py#L178
https://github.com/dwromero/ckconv
"""
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, Module
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import math

from tqdm.auto import tqdm


Linear = nn.Linear
sile = F.silu

import os,sys, importlib
source =  os.path.join(os.getcwd(), '..') # /home/tyk/train-box/torch_basic/.. 을 추가함
if source not in sys.path:
    sys.path.append(source)


"""
파이토치 셋업
"""
# 구버전 dropout 이상한거 처리
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    cudnn.benchmark = True
    

"""
reproducibility
- https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L31 참조
"""
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



"""
Arguments
"""
def argmunet():
    parser = argparse.ArgumentParser(decription='Single GPU Training')

    # Optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    # Scheduler
    # parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
    parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
    # Dataset
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
    # Dataloader
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--amp', type=str, default='fp16', choices=['fp16'])
    parser.add_argument('--check_out_dir', type=str, default='./checkpoint')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = argmunet()


    # Model class
    if args.model == 'resnet':
        # from model.resnet.res_models import Model
        model_cls = getattr(importlib.import_module('model.resnet.res_models'), "Model")
    elif args.model == 'vit':
        model_cls = getattr(importlib.import_module('model.transformer.vit'), "ViT")
            
    m = model_cls()
    
"""
dataset & dataloader
"""
def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.dataset == 'cifar10':
    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x : x.view(1, 1024).to())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data/cifar', train=True, download=True, transform=transform
    )
    train_set, _ = split_train_val(train_set, val_split=0.1)

    val_set = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform)
    _, val_set = split_train_val(val_set, val_split=0.1)

    test_set = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform)

    d_input = 3 if not args.grayscale else 1
    d_output = 10
    
elif args.dataset == 'mnist':
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
else:
    raise NotImplementedError


train_sampler = None
val_sampler = None
test_sampler = None

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           pin_memory=True,
                                           num_workers=2,
                                           sampler=train_sampler,
                                           drop_last=True,
                                           persistent_workers=True)

val_loader = torch.utils.data.DataLoader(val_set, 
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         pin_memory=True,
                                         num_workers=2,
                                         sampler=val_sampler,
                                         drop_last=False,)

test_loader = torch.utils.data.DataLoader(test_set, 
                                          batch_size=args.batch_size, 
                                          shuffle=False, 
                                          pin_memory=True,
                                          num_workers=2,
                                          sampler=test_sampler,
                                          drop_last=False,)


"""
Model
"""
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_checkpoint = args.act_ckpt

        self.layers = nn.ModuleList([])


        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        
    def forward(self, x, args):
        for layer in self.layers:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

    def register(self, name, tensor, lr=None):
        # 특정 파라미터가 별도의 lr로 학습되어야 하는 경우 사용
        if lr == 0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)
                

class Head(nn.Module):
    def 


# Init the model
g_running = StyledGenerator(code_size).cuda()
g_running.train(False)



"""
Optimizer
"""
def setup_optimizer(args, model, lr, weight_decay, epochs):
    # 레이어마다 
    all_parameters = list(model.parameters())

    # 특정 레이어에 _optim key라는게 있음 (보통은 없음)
    # 일반 레이어는 AdamW로 보낸다
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # 특정 레이어의 특정 파라마티의 attribute (_optim)에 optimizer 관련 정보 저장해줌
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]

    # 이 부분이 왜 굳이 있는지?
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]

    # list of dict를 looping하면서
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp} # weight_decay, lr을 넣음
        )

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    elif args.scheduler == 'reducedlr':
        if args.patience is None:
            raise ValueError("You must specify patience.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.2)
    else:
        raise NotImplementedError


    # Print optimizer info
    # optim이 있는 것만 key를 얻음 (아까 별로도 설정했던 것들에 대해 프린팅)
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler



"""
Loss
"""
# Option에 따른 다른 loss 사용
if args.loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss()
else:
    raise NotImplementedError
"""
train utils
"""
# Dynamic droptout : iteration이 돌때마다 dropout 값을 바꿀 수 있게끔
def set_dropout(model, new_p):
    for idx, m in enumerate(model.named_modules()):
        path = m[0]
        component = m[1]
        if isinstance(component, nn.Dropout):
            component.p = new_p

# Learning rate scheduling  from scratch
def adjust_lr(optimizer, lr):
  for group in optimizer:
    mult = group.get('mult', 1)
    group['lr'] = lr * mult

# dummy mixed precision : fp16 안켰을 때 이용
# 그냥 amp.autocast(enabled=...), enabled 사용하는게 편할듯
# https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/e95bcd46372573581ae8b34c083e65bd5e4e0e9e/src/worker.py#L10
class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exec_type, exc_value, traceback):
        return False


# data feeding conversion
def conversion(device, *items):
    return (data.to(device, pin_memory=True, non_blocking=True) for data in items)

# 만약에 체크포인트 파일이름에 step수가 표기되어 있을시 사용
# https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
def get_prev_step(args):
    if args.ckpt is not None:
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

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
        batch = conversion(device, *batch)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if args.fp16 else dummy_context_mgr():
            output = model(batch)
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            is_overflow = math.isnan(grad_norm)
            
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0:
            print(
                'Train epoch: {} [{}/{} (:.0f)%]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
                )
            )

@torch.no_grad()       
def test(args, model, device, dataloader, criterion, optimizer, scheduler, epoch, checkpoint=False):
    global best_acc # 외부에 있는 변수 가지고 와서 저장
    model.eval()
    eval_loss = 0 # 미비 배치 loss 총합
    correct = 0 # 미니 배치에서 correct 총합
    total = 0 # 미니 배치 총합
    acc = 0
    pbar = tqdm(enumerate(dataloader))

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = conversion(device, *(inputs, targets))
        outputs = model(inputs)
        # criterion은 batch-mean reduction 수행
        loss = criterion(outputs, targets)

        eval_loss += loss.item()
        # _, predicted = outputs.max(1)
        predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )

    acc = 100.*correct/total

    if checkpoint:
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'args': args,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            # 저장할 때 step이나 epoch을 체크포인트 이름에 넣어서 활용 가능
            torch.save(state, os.path.join(args.check_out_dir, f'ckpt_{str(epoch).zfill(6)}.pt'))
            best_acc = acc

    return acc
            
"""
training loop
- epoch단위로 할건지
"""

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


pbar = tqdm(range(start_epoch, args.epochs))


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
        
    train(args, model)
    val_acc = test(model, val_loader)
    test(model, test_loader)
    scheduler.step()


