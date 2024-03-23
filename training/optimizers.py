import torch

try:
    import bitsandbytes as bnb
    adam8bit_class = bnb.optim.Adam8bit
except ImportError:
    adam8bit_class = None
    # pass


optimizer_dict = {'adam': torch.optim.Adam, 'adam8bit': adam8bit_class, 'adamw': torch.optim.AdamW}