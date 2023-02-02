# -*- coding: utf-8 -*
import numpy as np
import random

import torch.nn.functional as F

import torch
from torch import nn

eps = np.finfo(np.float32).tiny


class IOULoss(nn.Module):

    default_hyper_params = dict(
        name="iou_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        # self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))
        self.t_one= torch.tensor(1., requires_grad=False).to(self.device)
        self.t_zero= torch.tensor(0., requires_grad=False).to(self.device)
        

        self.background = self.default_hyper_params["background"]
        self.ignore_label = self.default_hyper_params["ignore_label"]
        self.weight = self.default_hyper_params["weight"]

    def safelog(self, t):
        eps = np.finfo(np.float32).tiny

        t_eps = torch.tensor(eps, requires_grad=False).to(self.device)

        return torch.log(torch.max(t_eps, t))

    def forward(self, pred_box, target_box,target_label):
        pred = pred_box
        gt = target_box
        cls_gt = target_label
        mask = ((~(cls_gt == self.background)) *
                (~(cls_gt == self.ignore_label))).detach()
        mask = mask.type(torch.Tensor).squeeze(2).to(pred.device)

        aog = torch.abs(gt[:, :, 2] - gt[:, :, 0] +
                        1) * torch.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
        aop = torch.abs(pred[:, :, 2] - pred[:, :, 0] +
                        1) * torch.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

        iw = torch.min(pred[:, :, 2], gt[:, :, 2]) - torch.max(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = torch.min(pred[:, :, 3], gt[:, :, 3]) - torch.max(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = torch.max(iw, self.t_zero) * torch.max(ih, self.t_zero)

        union = aog + aop - inter
        iou = torch.max(inter / union, self.t_zero)
        loss = -self.safelog(iou)

        loss = (loss * mask).sum() / torch.max(
            mask.sum(), self.t_one) * self.default_hyper_params["weight"]
        iou = iou.detach()
        iou = (iou * mask).sum() / torch.max(mask.sum(), self.t_one)
        extra = dict(iou=iou)

        return loss,iou

def rpn_smoothL1(input, target, label, num_pos=16, ohem=None):
    '''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''
    loss_all = []
    for batch_id in range(target.shape[0]):
        min_pos = min(len(np.where(label[batch_id].cpu() == 1)[0]), num_pos)
        if ohem:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(
                    input[batch_id][pos_index], target[batch_id][pos_index], reduce=False)
                sort_index = torch.argsort(loss_bid.mean(1))
                loss_bid_ohem = loss_bid[sort_index[-num_pos:]]
            else:
                #loss_bid_ohem = torch.FloatTensor([0]).cuda()[0]
                loss_bid_ohem = torch.FloatTensor([0])[0]
            loss_all.append(loss_bid_ohem.mean())
        else:
            pos_index = np.where(label[batch_id].cpu() == 1)[0]
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss_bid = F.smooth_l1_loss(
                    input[batch_id][pos_index], target[batch_id][pos_index])
            else:
                loss_bid = torch.FloatTensor([0]).cuda()[0]
                #loss_bid = torch.FloatTensor([0])[0]
            loss_all.append(loss_bid.mean())
    final_loss = torch.stack(loss_all).mean()
    return final_loss


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


    criterion_reg = IOULoss()
    # print(pred_reg, gt_reg)
    loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)
    print(loss_reg)

