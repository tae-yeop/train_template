# data 구조
# root
# > id_1_0.jpg 
# > id_1_1.jpg
# > id_1_2.jpg  
# id 별로 관련 데이터를 랜덤하게 0,1,2 뽑아야 할 경우
# 이 경우 getitem에서 랜덤하게 뽑지 말자 => 미리 뽑아야할 전체 조합을 가지고 있어야 한다


# 다음처럼 하지 말기
class RandomDataset2(Dataset):
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        id_idx = idx // self.num_variations
        candidate = id_idx*self.num_variations + torch.randperm(self.num_variations)
        candidate = candidate[candidate!=idx]
        data1_sample = self.data[idx]
        data2_sample = self.data2[candidate[0]]

        return data1_sample, data2_sample