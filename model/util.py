'''
https://github.com/VITA-Group/Ultra-Data-Efficient-GAN-Training/blob/main/BigGAN%20and%20DiffAugGAN/utils/misc.py
'''
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as distributed

distributed.init_process_group(backend='nccl')

m = DistributedDataParallel(nn.Linear(10, 20).to('cuda'))

def toggle_grad(model, on, freeze_layers=-1):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        num_blocks = len(model.module.in_dims)
    else:
        num_blocks = len(model.is_dims)

    assert freeze_layers < num_blocks,"can't not freeze the {fl}th block > total {nb} blocks.".format(fl=freeze_layers, nb=num_blocks)
    if freeze_layers == -1:
        for name, param in model.named_parameters():
            param.requires_grad = on
    else:
        for name, param in model.named_parameters():
            param.requires_grad = on
        for layer in range(freeze_layers):
            block = "blocks.{layer}".format(layer=layer)
            if block in name:
                param.requires_grad = False