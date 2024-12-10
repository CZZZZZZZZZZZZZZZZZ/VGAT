import torch
import torch.nn as nn
import os
import numpy as np
import pdb


from models.PANTHER.file_util import save_pkl, load_pkl

class ProtoCount(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, proto_path='p',num = 10000):
        super().__init__()
       

        
        proto_path = proto_path

        if proto_path.endswith('pkl'):
            weights = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            weights = np.load(proto_path)

        self.n_proto = 16
        self.num = num
        self.prototypes = torch.from_numpy(weights).float()

        emb_dim = 1024

    def representation(self, x):
        """
        Compute the distance Eulcidean between prototypes and the patch features

        Args:
            x:

        Returns:

        """
        self.prototypes = self.prototypes.to(x.device)
        dist = torch.cdist(self.prototypes, x, p=2) # (1 x n_proto x n_instances)
        min_distances, _ = torch.min(dist, dim=0)  
        min_distances = min_distances.squeeze(0)  

        # 找到距离最短的num/2个实例的索引
        _, min_indices = torch.topk(min_distances, int(self.num/2), largest=False)

        # 找到距离最长的num/2个实例的索引
        _, max_indices = torch.topk(min_distances, int(self.num/2), largest=True)

        # 合并索引并去重
        combined_indices = torch.cat([min_indices, max_indices])
        unique_indices = torch.unique(combined_indices)
        if unique_indices.shape[0] < self.num:
        # 如果有重复，需要补充更多的实例
            remaining = self.num - unique_indices.shape[0]
            all_indices = torch.arange(4*self.num)
            available_indices = torch.tensor(list(set(all_indices.tolist()) - set(unique_indices.tolist())))
            additional_indices = available_indices[torch.randperm(available_indices.shape[0])[:remaining]]
            unique_indices = torch.cat([unique_indices, additional_indices])

        # 选择相应的实例
        x = x[unique_indices]  
        return x

    def forward(self, x):
        out = self.representation(x)
        return out
    
    