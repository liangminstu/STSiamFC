import math

import torch
from torch import nn, Tensor
from typing import Optional

from data_process.gauss import gaussian_label_function
from network.multiattention import MultiheadAttention


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())



class FeatureDescend(nn.Module):
    def __init__(self,d_model=128):
        super().__init__()
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        # Implementation of Feedforward model
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        batch, dim,num_imgs, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, sfeats, tfeats, pos: Optional[Tensor] = None):
        #adjust dim
        #searh
        # sfeat_shape = sfeat.shape
        batch, dim, num_imgs, hs, ws = sfeats.shape
        input_shape = sfeats.shape
        sfeats = sfeats.reshape(batch, dim, num_imgs, -1).permute(2, 3, 0, 1)
        sfeats = sfeats.reshape(-1, batch, dim)


        # template
        batch, dim, num_imgt, ht, wt = tfeats.shape
        tfeats = tfeats.reshape(batch, dim, num_imgt, -1).permute(2, 3, 0, 1)
        tfeats = tfeats.reshape(-1, batch, dim)

        # template-mask
        pos = pos.view(batch, 1, num_imgt, -1).permute(2, 3, 0, 1)
        pos = pos.reshape(-1, batch, 1)
        # print(sfeats.shape,tfeats.shape,pos.shape)

        # cross-attention

        # S(mask)=A(t-s)M'*S
        # print(sfeat.shape)
        mask = self.cross_attn(query=sfeats, key=tfeats, value=pos)
        tgt2 = sfeats * mask
        tgt2 = self.instance_norm(tgt2, input_shape)
        # S(feat)=A(t-s)(T'*M')+S
        tgt3 = self.cross_attn(query=sfeats, key=tfeats, value=tfeats * pos)
        tgt4 = sfeats + tgt3
        tgt4 = self.instance_norm(tgt4, input_shape)

        tgt = tgt2 + tgt4
        tgt = self.instance_norm(tgt, input_shape)
        tgt = tgt.reshape(num_imgs,  hs, ws, batch, dim).permute(3,4,0,1,2)


        return tgt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    target_bb = torch.tensor([[10,20,50,50],[30,50,100,100],[100,100,120,120]])

    # output_sigma = settings.output_sigma_factor / settings.search_area_factor
    output_sigma = 1 / 4 / 6
    gauss_label = gaussian_label_function(target_bb, output_sigma, 1, 15, 127, end_pad_if_even=True, density=False,
                                          uni_bias=0)


    template = torch.randn(1, 128, 3, 15, 15)#.reshape(1, 256 , 3 , -1).permute(2,3,0,1)


    search = torch.randn(1, 128, 3, 31, 31)#.reshape(1, 256 , 1 , -1).permute(2,3,0,1)


    FEAT=FeatureDescend()

    print(FEAT(search,template,gauss_label).shape)
