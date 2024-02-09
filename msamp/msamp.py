import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import msamp

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, targe = 






from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import argparse

def main():
    batch_size = 64
    test_batch_size = 1000
    epcohs = 4
    enable-msamp = True
    opt-level = 'O1'
    seed = 1
    # Learning rate step gamma
    gamma = 0.7
    use_cuda = True
    torch.manual_seed(seed)
    device = torch.device('cuda')

    train_kwargs = {'batch_size' : batch_size}
    test_kwargs = {'batch_size' : test_batch_size}

    if use_cuda:
        cuda_kwargs = {}
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose()
    dataset1 = datasets.MNIST()
    dataset2 = datasets.MNIST()
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().

    if args.enable_msamp:
        model, optimizer = msamp.initialize(model, optimizer, opt_level=opt_level)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    main()
    