
import torch
from torch.utils.data import Dataset

import os
from PIL import Image


class MyImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.file_list = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)]
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        img = Image.open(file).convert('RGB')

        # 흑백을 3차원처럼 처리
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim=0)
        if self.transforms is not None:
            img = self.transforms(img)

        return img


import os
import os.path
class FolderClassDataset(Dataset):
    """
    폴더별로 클래스 이미지들이 나눠져 있을 때
    """
    def __init__(self, root):
        # scandir : root 바로 아래에 있는 폴더와 파일 리스트를 준다
        # is_dir : 디렉토리만
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir()) 
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {root}")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}



import re
class FileNameClassDataset(Dataset):
    """
    이미지 파일 이름에 클래스가 있을 경우 : cat_123.jpg
    미리 label_to_id 리스트를 들고 있어야 함
    """
    def __init__(self, root, image_transform=None, label_to_id=None):
        self.root = root
        self.file_list = [os.path.join(self.root, file_path) for file_path in os.listdir(self.root)]
        self.image_transform = image_transform
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            self.label_to_id = {'cat' : 0, 'dog': 1}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        image = Image.open(fname).convert("RGB")
       
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = self.extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}

    def extract_label(fname):
        # '/home/tyk/MS-AMP' => ['', 'home', 'tyk', 'MS-AMP']
        stem = fname.split(os.path.sep)[-1]
        # cat_1234.jpg => cat
        return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]

