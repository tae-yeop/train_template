"""
분산 처리를 위한 샘플러
"""
import torch.distributed as dist
import math
import torch
from torch.utils.data.distributed import Sampler

class CustomDistributedSampler(Sampler):
    # 테스트용인지 flag 추가
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, is_test=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.is_test = is_test

        if self.is_test:
            # 전체 데이터수를 world_size로 나눈다 => 랭크별 담당해야할 데이터 갯수
            self.divided = int(len(self.dataset) // self.num_replicas)
            # 안나눠 떨어지는 나머지 데이터 갯수
            remainder = len(self.dataset) % self.num_replicas
            # 랭크별 담당해야할 샘플수, 마지막이면 나머지도 같이
            self.num_samples = self.divided + remainder if self.rank == (self.num_replicas-1) else self.divided
            self.total_size = len(self.dataset)

        else: # 학습일떄
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas

        self.shuffle = shuffle
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.is_test:
            # add extra samples to make it evenly divisible
      indices += indices[:(self.total_size - len(indices))]
      assert len(indices) == self.total_size
        # subsample
        if self.is_test:
            if self.rank == (self.num_replicas-1):
                # 마지막 랭크라면 끝까지 모두 담당
                indices = indices[self.rank * self.divided : self.total_size]
            else:
                # 자기 랭크 몫만 담당
                indices = indices[self.rank * self.divided : (self.rank + 1) * self.divided]
        else: # 학습시 
            # 만약 num_replicas=3이면  [0, 1, 2, 0, 1, 2, ....]순으로
            # 각자가 데이터를 찜한다
            indices = indices[self.rank:self.total_size:self.num_replicas]
      assert len(indices) == self.num_samples

        if isinstance(self.dataset, Sampler):
            orig_indicies = list(iter(self.dataset))
            indicies = [orig_indices[i] for i in indices]

        return iter(indicies)