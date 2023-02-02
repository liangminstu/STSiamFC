# -*- coding: utf-8 -*
from builtins import print
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


# from losses.module import ModuleBase


eps = np.finfo(np.float32).tiny


class SigmoidCrossEntropyCenterness(nn.Module):
    default_hyper_params = dict(
        name="centerness",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self):
        super(SigmoidCrossEntropyCenterness, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.t_one = torch.tensor(1., requires_grad=False).to(self.device)
        self.background = self.default_hyper_params["background"]
        self.ignore_label = self.default_hyper_params["ignore_label"]
        self.weight = self.default_hyper_params["weight"]
    
    def safelog(self,t):
        eps = np.finfo(np.float32).tiny

        t_eps = torch.tensor(eps, requires_grad=False).to(self.device)

        return torch.log(torch.max(t_eps, t))


    def forward(self, pred_data, target_data):
        r"""
        Center-ness loss
        Computation technique originated from this implementation:
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        P.S. previous implementation can be found at the commit 232141cdc5ac94602c28765c9cf173789da7415e

        Arguments
        ---------
        pred: torch.Tensor
            center-ness logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data
        label = target_data
        
        mask = (~(label == self.background)).type(torch.Tensor).to(pred.device)

        not_neg_mask=(pred>=0).type(torch.Tensor).to(pred.device)
        
        
        loss=(pred*not_neg_mask-pred*label + self.safelog(1.+torch.exp(-torch.abs(pred))))*mask
        
        loss_residual=(-label*self.safelog(label)-(1-label)*self.safelog(1-label))*mask
                                  reduction="none") * mask
        loss = loss - loss_residual.detach()

        
        loss = loss.sum() / torch.max(mask.sum(), self.t_one) * self.weight
      

        return loss

if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))


    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)

    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))

    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))


    criterion_ctr = SigmoidCrossEntropyCenterness()
    loss_ctr = criterion_ctr.forward(pred_ctr, gt_ctr)
    print(loss_ctr)


  
