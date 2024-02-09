import torch
import torch.nn as nn
from torch.optim import SGD

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(512, 512, bias=False)
        self.w2 = nn.Linear(512, 1, bias=False)
    
    def forward(self, x):
        z1 = self.w1(x)
        z2 = self.w2(z1)
        return z2

fp32_model= Net().to("cuda")
optimizer = SGD(fp32_model.parameters(), lr=1e-2)

f"GPU = {torch.cuda.memory_allocated(0) / (1024 ** 2)} GiB"


# 1) Float2Half

fp16_model = Net().half().to("cuda")
fp16_model.load_state_dict(fp32_model.state_dict())


f"GPU = {torch.cuda.memory_allocated(0) / (1024 ** 2)} GiB"


# 2) Forward