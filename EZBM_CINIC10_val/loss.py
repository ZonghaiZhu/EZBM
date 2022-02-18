# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://www.jianshu.com/p/0c159cdd9c50


class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = F.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MixCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean', ):
        super(MixCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target, importance):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits * importance

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


