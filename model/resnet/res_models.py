import torch
import torch.nn as nn

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