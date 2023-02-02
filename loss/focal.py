# -*- coding: utf-8 -*

from builtins import print
from typing import Type
import torch
from torch import device, nn
import numpy as np


class SigmoidCrossEntropyRetina(nn.Module):

    default_hyper_params = dict(
        name="focal_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.75,
        gamma=2.0,
    )

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.t_one= torch.tensor(1., requires_grad=False).to(self.device)
        self.gamma=torch.tensor(float(self.default_hyper_params["gamma"]),
                     requires_grad=False).to(self.device)
        self.alpha=self.default_hyper_params["alpha"]
        self.weight = self.default_hyper_params["weight"]
        
        
    def safelog(self,t):
        eps = np.finfo(np.float32).tiny

        t_eps = torch.tensor(eps, requires_grad=False).to(self.device)

        return torch.log(torch.max(t_eps, t))


    def forward(self, pred_data, target_data):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
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
       
        mask = ~(label == self.default_hyper_params["ignore_label"])
      

        mask = mask.type(torch.Tensor).to(label.device)

        vlabel = label * mask

        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2] + 1)

        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2] + 1)
        index_mat = vlabel.type(torch.LongTensor)
        onehot_ = zero_mat.scatter(2, index_mat, one_mat)
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)
        ####

        pred = torch.sigmoid(pred)
        
        # pred = (pred - pred.min()) / (pred.max() - pred.min())
       


        pos_part = (1 - pred)** self.gamma * onehot * self.safelog(pred)
        ####
        neg_part = pred ** self.gamma * (1 - onehot) * self.safelog(1 - pred)
        loss = -(self.alpha * pos_part +
                 (1 - self.alpha) * neg_part).sum(dim=2) * mask.squeeze(2)

        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)


        loss = loss.sum() / torch.max(positive_mask.sum(), self.t_one) * self.weight


        return loss

if __name__ == '__main__':
    B = 2
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    print(pred_cls)

    #随机产生0或1的值
    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)

    criterion_cls = SigmoidCrossEntropyRetina()
    loss_cls = criterion_cls.forward(pred_cls, gt_cls)
    print(loss_cls)
