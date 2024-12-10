import numpy as np

import torch
import torch.nn as nn
import pickle
import random


from models.PANTHER.layers import PANTHERBase
from models.PANTHER.util import initialize_weights






class PANTHER(nn.Module):
    def __init__(self, n_classes=4, fusion="concat", model_size="small",proto_path = 'p'):
        super(PANTHER, self).__init__()

       
        self.n_classes = n_classes
        self.fusion = fusion
        self.proto_path = proto_path

        ###
        self.panther = PANTHERBase(1024,proto_path = self.proto_path)
        
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        
        
        self.ly1 = nn.Linear(32784, 4096)
        self.relu1 = torch.nn.ReLU()
        self.ly2 = nn.Linear(4096, 1024)
        self.relu2 = torch.nn.ReLU()
        self.ly3 = nn.Linear(1024, 256)
        self.relu3 = torch.nn.ReLU()
        self.classifier = nn.Linear(256, self.n_classes)

        self.apply(initialize_weights)
    
    
    def forward(self, **kwargs):
        # meta genomics and pathomics features
        # 假设 x_path 是传递给函数的包含特征的 n x 1024 维张量
        x_path = kwargs["x_path"]  # 示例，x_path 应该是 n x 1024 的张量

        
        out, qqs = self.panther(x_path.unsqueeze(0))
       
        out = self.ly1(out)
        out = self.relu1(out)
        out = self.ly2(out)
        out = self.relu2(out)
        out = self.ly3(out)
        out = self.relu3(out)
        # predict
        logits = self.classifier(out)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S