import torch.nn.functional as F
import torch.nn as nn

from src.arch.module import MaxoutBackbone, VGG11BNBackbone, Resnet18Backbone, EmbeddingBackbone, PredictHead, VniNetBackbone


# Recognition
class Maxout(nn.Module):
    def __init__(self, num_classes):
        super(Maxout, self).__init__()
        self.backbone = MaxoutBackbone()
        self.classifier = PredictHead(num_classes, 256)

    def forward(self, input):
        feature = self.backbone(input)
        feature = F.normalize(feature)
        prediction = self.classifier(feature)
        return {'feature': feature, 'prediction': prediction}


class Embedding(nn.Module):
    def __init__(self, num_classes):
        super(Embedding, self).__init__()
        self.backbone = EmbeddingBackbone()
        self.classifier = PredictHead(num_classes, 256)

    def forward(self, input):
        feature = self.backbone(input)
        feature = F.normalize(feature)
        prediction = self.classifier(feature)
        return {'feature': feature, 'prediction': prediction}


class VniNet(nn.Module):
    def __init__(self, num_classes):
        super(VniNet, self).__init__()
        self.backbone = VniNetBackbone()
        self.classifier = PredictHead(num_classes, 256)

    def forward(self, input):
        feature = self.backbone(input)
        feature = F.normalize(feature)
        prediction = self.classifier(feature)
        return {'feature': feature, 'prediction': prediction}


class VGG11BN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG11BN, self).__init__()
        self.backbone = VGG11BNBackbone(pretrained)
        self.classifier = PredictHead(num_classes, 512)

    def forward(self, input):
        feature = self.backbone(input)
        feature = F.normalize(feature)
        prediction = self.classifier(feature)
        return {'feature': feature, 'prediction': prediction}


class Resnet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Resnet18, self).__init__()
        self.backbone = Resnet18Backbone(pretrained)
        self.classifier = PredictHead(num_classes, 512)

    def forward(self, input):
        feature = self.backbone(input)
        feature = F.normalize(feature)
        prediction = self.classifier(feature)
        return {'feature': feature, 'prediction': prediction}


if __name__ == "__main__":
    pass
