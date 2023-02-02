import torch
from torch import nn

from losses.ctr import SigmoidCrossEntropyCenterness
from losses.focal import SigmoidCrossEntropyRetina
from losses.iouloss import rpn_smoothL1, IOULoss


class TotalLoss(nn.Module):

    def __init__(self):
        super(TotalLoss, self).__init__()

    def forward(self, pred_cls, pred_ctr,pred_reg,gt_cls, gt_ctr, gt_reg):
    
        #cls-loss
        criterion_cls = SigmoidCrossEntropyRetina()
        loss_cls = criterion_cls(pred_cls, gt_cls)

        #ctr-loss
        criterion_ctr = SigmoidCrossEntropyCenterness()
        loss_ctr = criterion_ctr(pred_ctr, gt_ctr)

        # #reg-loss
        criterion_reg = IOULoss()
        loss_reg,iou = criterion_reg(pred_reg, gt_reg, gt_cls)
        loss = (loss_cls + loss_ctr+ 3*loss_reg)/pred_cls.shape[0]
        
        
        return loss,iou
