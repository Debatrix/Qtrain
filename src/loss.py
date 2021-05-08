from src.framework import SegPredModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1),
                               -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(
                -1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()).to(input.device)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at).to(input.device)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) +
                                                       smooth)
        score = 1 - score.sum() / num
        return score


# ##############################################################################


class SoftMLoss(nn.Module):
    def __init__(self, gamma=0.8, mask_loss='sl1', pred_loss='mse') -> None:
        super(SoftMLoss, self).__init__()
        self.gamma = gamma

        if pred_loss.lower() == 'sl1':
            self.pred_loss = nn.SmoothL1Loss()
        elif pred_loss.lower() == 'l1':
            self.pred_loss = nn.L1Loss()
        else:
            self.pred_loss = nn.MSELoss()

        if mask_loss.lower() == 'sl1':
            self.mask_loss = nn.SmoothL1Loss()
        elif mask_loss.lower() == 'l1':
            self.mask_loss = nn.L1Loss()
        else:
            self.mask_loss = nn.MSELoss()

    def forward(self, input):
        pred_loss = self.pred_loss(input['dfs'], input['pdfs'])
        mask_loss = self.mask_loss(input['pmask'], input['mask'])
        loss = self.gamma[0] * mask_loss + self.gamma[1] * pred_loss
        return {
            'loss': loss,
            'pred_loss': pred_loss.cpu().item(),
            'mask_loss': mask_loss.cpu().item()
        }


class HardMLoss(nn.Module):
    def __init__(self, gamma=0.8, mask_loss='ce', pred_loss='mse') -> None:
        super(HardMLoss, self).__init__()
        self.gamma = gamma

        if pred_loss.lower() == 'sl1':
            self.pred_loss = nn.SmoothL1Loss()
        elif pred_loss.lower() == 'l1':
            self.pred_loss = nn.L1Loss()
        else:
            self.pred_loss = nn.MSELoss()

        if mask_loss.lower() == 'focal':
            self.mask_loss = FocalLoss()
        elif mask_loss.lower() == 'dice':
            self.mask_loss = SoftDiceLoss()
        else:
            self.mask_loss = nn.CrossEntropyLoss()

    def forward(self, input):
        pred_loss = self.pred_loss(input['dfs'], input['pdfs'])
        mask_loss = self.mask_loss(input['pmask'], input['mask'])
        loss = self.gamma[0] * mask_loss + self.gamma[1] * pred_loss
        return {
            'loss': loss,
            'pred_loss': pred_loss.cpu().item(),
            'mask_loss': mask_loss.cpu().item()
        }


class PDFSLoss(nn.Module):
    def __init__(self, pred_loss='mse', **kargs) -> None:
        super(PDFSLoss, self).__init__()

        if pred_loss.lower() == 'sl1':
            self.pred_loss = nn.SmoothL1Loss()
        elif pred_loss.lower() == 'l1':
            self.pred_loss = nn.L1Loss()
        else:
            self.pred_loss = nn.MSELoss()

    def forward(self, input):
        pred_loss = self.pred_loss(input['dfs'], input['pdfs'])
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss.cpu().item(),
            'mask_loss': -1
        }
