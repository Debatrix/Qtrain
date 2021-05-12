import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 loss_type='arcface',
                 eps=1e-7,
                 s=None,
                 m=None):
        '''
        Author:cvqluu
        Url: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch

        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) -
                                  self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]),
                                -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]),
                            -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([
            torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0)
            for i, y in enumerate(labels)
        ],
                         dim=0)
        denominator = torch.exp(numerator) + torch.sum(
            torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


# ##############################################################################


class PredictLoss(nn.Module):
    def __init__(self, pred_loss='ce', *args, **kargs) -> None:
        super(PredictLoss, self).__init__()

        if pred_loss.lower() == 'ce':
            self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        elif pred_loss.lower() == 'arcface':
            self.pred_loss = AngularPenaltySMLoss(loss_type='arcface')
        elif pred_loss.lower() == 'sphereface':
            self.pred_loss = AngularPenaltySMLoss(loss_type='sphereface')
        elif pred_loss.lower() == 'cosface':
            self.pred_loss = AngularPenaltySMLoss(loss_type='cosface')
        else:
            raise ValueError('Unsupported Loss: ' + pred_loss)

    def forward(self, input):
        pred_loss = self.pred_loss(input['prediction'], input['label'])
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss,
        }


class SiameseLoss(nn.Module):
    def __init__(self,
                 pred_loss='ce',
                 feat_loss='cos',
                 margin=0.0,
                 alpha=1) -> None:
        super(SiameseLoss, self).__init__()
        if feat_loss.lower() == 'cos':
            self.feat_loss = nn.CosineEmbeddingLoss(margin=margin,
                                                    reduction='sum')
        else:
            raise ValueError('Unsupported Loss: ' + feat_loss)

    def _train(self, input):

        feat_loss = self.feat_loss(
            input['feature1'],
            input['feature2'],
            input['flag'],
        )

        return {
            'loss': feat_loss,
            'feat_loss': feat_loss,
        }

    def _val(self, input):
        return {}

    def forward(self, input):
        if 'flag' in input:
            loss = self._train(input)
        else:
            loss = self._val(input)

        return loss