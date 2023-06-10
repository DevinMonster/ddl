import torch
from torch import nn
from torch.functional import F


class CrossEntropyLoss(nn.Module):
    '''
    traditional CELoss for multi-class classify
    '''

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        '''
        :param x: model's logits with shape {B,C(1+old+new),H,W}
        :param y: true label with shape {B,H,W}
        '''
        return F.nll_loss(F.log_softmax(x, dim=1), y,
                          reduction=self.reduction)


class UnbiasedCrossEntropyLoss(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        return F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)
