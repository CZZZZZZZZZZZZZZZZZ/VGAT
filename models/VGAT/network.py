import numpy as np

import torch
import torch.nn as nn
import pickle
import random

from .util import initialize_weights
from .nystrom_attention import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention
from models.PANTHER.layers import PANTHERBase
from .cluster_protocount import ProtoCount


from .model_reconstruction import Reconstruction_Net


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head_fusion='mean'):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            return_attn= True,
            head_fusion='max',
        )

    def forward(self, x):
        _, attn = self.attn(self.norm(x))
        x = x + _
        return x, attn


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h,_ = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h,attn = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:],attn[:,:H+1]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h,_ = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h,_ = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class VGAT(nn.Module):
    def __init__(self, n_classes=4, fusion="concat", model_size="small",proto_path = 'p',num=10000,select = None):
        super(VGAT, self).__init__()

       
        self.n_classes = n_classes
        self.fusion = fusion
        self.proto_path = proto_path
        self.num = num
        self.select = select

        ###
        self.panther = PANTHERBase(1024,proto_path = self.proto_path)
        self.cluster = ProtoCount(proto_path = self.proto_path,num = self.num)
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        
        
        self.recon_net = Reconstruction_Net(omic_sizes=[256], num_tokens=1)
        
        

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
       

        
        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        
        
       
       
        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)
    
    def EM(self,x,num):
        
        x_path = x
        original_x_path = x_path

        # qqs is patchs probs of each cluster
        p = x_path
        _, qqs = self.panther(p.unsqueeze(0))

        probs = qqs
        probs = probs.squeeze(dim=0).squeeze(dim=-1)

        # Determine classification and corresponding probabilities
        max_probs, patch_classes_indexes = torch.max(probs, dim=1)
        patch_classes = patch_classes_indexes + 1

        # create map to origin index
        patch_map = list(zip(range(len(probs)), patch_classes.tolist(), max_probs.tolist()))

        # calculate the number of patches in each class
        num_classes = 16
        counts = torch.bincount(patch_classes, minlength=num_classes+1)[1:]

        selected_indices = []

        # Identify classes with fewer than num/32 patches and retrieve the indices of all patches for those classes.
        for category, count in enumerate(counts, start=1):
            if count < num/32:
                for original_index, patch_class, probability in patch_map:
                    if patch_class == category:
                        selected_indices.append(original_index)

        # "For classes with patch counts not less than num/16, retrieve the indices of the highest num/32 and lowest num/32 patches based on probabilities."
        for category, count in enumerate(counts, start=1):
            if count >= num/16:
                n = int(num/32)
                category_patches = [patch for patch in patch_map if patch[1] == category]
                sorted_category_patches = sorted(category_patches, key=lambda x: x[2])
                selected_indices.extend([patch[0] for patch in sorted_category_patches[:n]])  # 概率最低的 50 个
                selected_indices.extend([patch[0] for patch in sorted_category_patches[-n:]])  # 概率最高的 50 个

        # non-overlap to num patchs
        if len(selected_indices) < num:
            remaining_indices = [patch[0] for patch in patch_map if patch[0] not in selected_indices]
            random.shuffle(remaining_indices)
            selected_indices.extend(remaining_indices[:num - len(selected_indices)])

        
        selected_indices_tensor = torch.tensor(selected_indices)
        result = original_x_path[selected_indices_tensor]
            
        return result

       
        
        
    def forward(self, **kwargs):
        
        x_path = kwargs["x_path"]  
        
        select = self.select
        num = self.num
        
        
        if x_path.size(0) > num:
            if select =='em':
                h = self.EM(x_path,num)
            elif select =='rand':
                h = x_path[np.random.choice(x_path.size(0), num, replace=False)]
            elif select =='cluster':
                h = self.cluster(x_path)
            else:
                h = x_path
        else:
            h = x_path
            
        re_x, _, _ = self.recon_net(x_path=h)
        
        x = re_x[0]
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
       
        
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        
        
        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder,attn = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
       
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            x)  # cls token + patch tokens
        
        
        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        cls_token_pathomics_encoder,
                        cls_token_genomics_encoder,
                        
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(
                cls_token_pathomics_encoder,
                cls_token_genomics_encoder,
                
            )  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        # predict
        logits = self.classifier(fusion)  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S,re_x,attn