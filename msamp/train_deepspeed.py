import argparse
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from msamp import deepspeed

def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    parser.add_argument()
    parser.add_argument()

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=30, type=int)
    parser.add_argument('--local-interval', default=200, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

add_argument()