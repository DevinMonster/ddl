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
    '''
    this is a direct implement of MiB's UnbiasedCE to
    deal with the old classes become to background at
    current stage
    '''

    def __init__(self, num_old_classes, reduction='mean'):
        super().__init__()
        self.n_old = num_old_classes
        self.reduction = reduction

    def forward(self, x, y):
        '''
        :param x: model's logits with shape B,C(1+old+new),H,W
        :param y: true label with shape B,H,W
        '''
        n_old = self.n_old
        p_x = F.softmax(x, dim=1)
        p_x[:, 0] = torch.sum(p_x[:, :n_old], dim=1)
        p_x[:, 1:n_old] = 0.

        # if current pixel belongs to old classes make it to background class
        labels = y.clone()
        labels[y < n_old] = 0
        return F.nll_loss(torch.log(p_x), labels, reduction=self.reduction)


class KDLoss(nn.Module):
    '''
    traditional Knowledge Distillation Loss which
    can transfer old model's knowledge to new one
    '''

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        '''
        :param x: new model's output logits with shape {B,C1(1+old+new),H,W}
        :param y: old model's output logits with shape {B,C2(1+old),H,W}
        '''
        c2 = y.shape[1]
        # drop new classes logits
        x = x.narrow(1, 0, c2)
        # q^{t-1}_x
        q_prev = F.softmax(y, dim=1)
        # log{q^t_x}
        log_q_cur = F.log_softmax(x)
        # B,H,W
        loss = -((q_prev * log_q_cur).mean(dim=1))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class UnbiasedKDLoss(nn.Module):
    '''
    this is a direct implement of MiB's unbiased
    knowledge distillation utils that can fix the issue
    if current classes' pixel belongs to previous stage's background
    '''

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        '''
        :param x: new model's output logits with shape {B,C1(1+old+new),H,W}
        :param y: old model's output logits with shape {B,C2(1+old),H,W}
        '''
        c1, c2 = x.shape[1], y.shape[1]
        assert c1 > c2, f"must c1 > c2"
        # q^{t-1}_x
        q_prev = F.softmax(y, dim=1)
        # log{q^t_x}
        q_cur = F.softmax(x, dim=1)
        # sum up all the probability new classes to background class
        idx = torch.tensor([0] + list(range(c2, c1))).to(x.device)
        # add up new classes prob to background classes
        q_cur[:, 0] = torch.sum(q_cur[:, idx], dim=1)
        log_q_cur = torch.log(q_cur.narrow(1, 0, c2))
        # utils
        assert log_q_cur.shape == q_prev.shape
        loss = -((q_prev * log_q_cur).mean(dim=1))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class MiBLoss(nn.Module):
    '''
    simple implement of Modeling the Background Loss
    '''

    def __init__(self, l=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.l = l

    def forward(self, x_old, x_new, y):
        new_cls = x_new.shape[1] - x_old.shape[1]
        uce = UnbiasedCrossEntropyLoss(new_cls, self.reduction)
        ukd = UnbiasedKDLoss(self.reduction)
        return uce(x_new, y) + self.l * ukd(x_new, x_old)
