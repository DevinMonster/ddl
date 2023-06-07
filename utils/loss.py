import torch
from torch import nn
from torch.functional import F


class CrossEntropyLoss(nn.Module):
    '''
    traditional CELoss for multi-class classify
    '''

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, x, y):
        '''
        :param x: model's logits with shape {B,C(1+old+new),H,W}
        :param y: true label with shape {B,H,W}
        '''
        return F.nll_loss(F.log_softmax(x, dim=1), y,
                          reduction=self.reduction, ignore_index=self.ignore_index)


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
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class LocalPODLoss(nn.Module):
    '''
    direct implementation of Local POD Distillation loss
    '''

    def __init__(self, S=2, alpha=1.):
        super().__init__()
        self.S = S
        self.alpha = alpha

    @staticmethod
    def POD(x, w, h):
        left = torch.sum(x, dim=2)
        right = torch.sum(x, dim=3)
        return torch.cat([left / w, right / h], dim=1)

    def local_pod(self, h_n, h_o):
        W, H = h_n.shape[-2], h_n.shape[-1]
        P_n, P_o = None, None
        d = 1
        for s in range(self.S):
            w, h = W // d, H // d
            if w == 0 or h == 0: break
            for i in range(0, W - w):
                for j in range(0, H - h):
                    p_n = self.POD(h_n[:, :, i:i + w, j:j + h], w, h)
                    p_o = self.POD(h_o[:, :, i:i + w, j:j + h], w, h)
                    if P_n is None:
                        P_o, P_n = p_o, p_n
                    else:
                        P_o = torch.cat([P_o, p_o], dim=1)
                        P_n = torch.cat([P_n, p_n], dim=1)
            d *= 2
        return torch.norm(P_n - P_o, 2)

    def forward(self, new_f, old_f):
        loss = torch.tensor(1e-6)
        for key in new_f.keys():
            h_n = new_f[key]
            h_o = old_f[key]
            loss += self.local_pod(h_n, h_o)
        return self.alpha * loss / (len(new_f) + 1)


class MiBLoss(nn.Module):
    '''
    simple implement of Modeling the Background Loss
    '''

    def __init__(self, old_cls, alpha=1., reduction='mean'):
        super().__init__()
        # self.uce = UnbiasedCrossEntropyLoss(old_cls, reduction)
        self.uce = UnbiasedCrossEntropyLoss(old_cls)
        self.ukd = UnbiasedKDLoss(reduction, alpha)

    def forward(self, y_new, y, y_old=None):
        new_model_loss = self.uce(y_new, y)
        distill_loss = self.ukd(y_new, y_old) if y_old is not None else 1e-6
        return new_model_loss + distill_loss
