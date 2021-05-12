from src.arch.module import se_resnet18
from src.framework import Module
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


class VGG16BN(nn.Module):
    def __init__(self, bg_classifier='none', pretrained=True):
        super(VGG16BN, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.vgg16_bn(pretrained)

        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512,
                                    out_features=out_ch,
                                    bias=True)

        self.features[0] = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_in')
        init.constant_(self.classifier.bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, 512)
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class VGG11BN(nn.Module):
    def __init__(self, bg_classifier='none', pretrained=True):
        super(VGG11BN, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.vgg11_bn(pretrained)

        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512,
                                    out_features=out_ch,
                                    bias=True)

        self.features[0] = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_in')
        init.constant_(self.classifier.bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, 512)
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class Resnet101(nn.Module):
    def __init__(self, bg_classifier='none', drop=False, pretrained=True):
        super(Resnet101, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.resnet101(pretrained)

        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if drop:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=2048, out_features=out_ch, bias=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=out_ch, bias=True), )

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier[-1].weight, a=0, mode='fan_in')
        init.constant_(self.classifier[-1].bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class Resnet50(nn.Module):
    def __init__(self, bg_classifier='none', drop=False, pretrained=True):
        super(Resnet50, self).__init__()
        self.drop = drop
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.resnet50(pretrained)

        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=2048,
                                    out_features=out_ch,
                                    bias=True)

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_in')
        init.constant_(self.classifier.bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class Resnet34(nn.Module):
    def __init__(self, bg_classifier='none', pretrained=True):
        super(Resnet34, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.resnet34(pretrained)

        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(in_features=512,
                                    out_features=out_ch,
                                    bias=True)

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_in')
        init.constant_(self.classifier.bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class Resnet18(nn.Module):
    def __init__(self, bg_classifier='none', drop=False, pretrained=True):
        super(Resnet18, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.resnet18(pretrained)

        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if drop:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=512, out_features=out_ch, bias=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512, out_features=out_ch, bias=True), )

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier[-1].weight, a=0, mode='fan_in')
        init.constant_(self.classifier[-1].bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class MobileNet(nn.Module):
    def __init__(self, bg_classifier='none', pretrained=True):
        super(MobileNet, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = torchvision.models.mobilenet_v2(pretrained)

        self.features = model.features
        self.features[0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=out_ch, bias=True),
        )

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')
        init.kaiming_normal_(self.classifier[-1].weight, a=0, mode='fan_in')
        init.constant_(self.classifier[-1].bias, 0)

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class Inception(torchvision.models.Inception3):
    def __init__(self, bg_classifier='none', pretrained=True):
        super(Inception, self).__init__(num_classes=1000,
                                        aux_logits=False,
                                        transform_input=False)
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        if pretrained:
            self.load_state_dict(model_zoo.load_url(url), strict=False)
        import scipy.stats as stats

        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        self.Conv2d_1a_3x3 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32, eps=0.001),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(2048, out_ch)

        X = stats.truncnorm(-2, 2, scale=0.1)
        values = torch.Tensor(X.rvs(self.Conv2d_1a_3x3[0].weight.numel()))
        values = values.view(self.Conv2d_1a_3x3[0].weight.size())
        self.Conv2d_1a_3x3[0].weight.data.copy_(values)

        nn.init.constant_(self.Conv2d_1a_3x3[1].weight, 1)
        nn.init.constant_(self.Conv2d_1a_3x3[1].bias, 0)

        X = stats.truncnorm(-2, 2, scale=0.1)
        values = torch.Tensor(X.rvs(self.fc.weight.numel()))
        values = values.view(self.fc.weight.size())
        self.fc.weight.data.copy_(values)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        feature = self.avgpool(x)
        feature = torch.flatten(feature, 1)

        prediction = F.dropout(feature, training=self.training)
        # prediction = torch.flatten(prediction, 1)
        prediction = self.fc(prediction)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


class SEnet18(Module):
    def __init__(self, bg_classifier='none', drop=False, pretrained=True):
        super(SEnet18, self).__init__()
        self.bg_classifier = bg_classifier
        if bg_classifier == 'integration':
            out_ch = 3
        elif bg_classifier == 'independence':
            out_ch = 4
        else:
            out_ch = 2

        model = se_resnet18(out_ch)

        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if drop:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=512, out_features=out_ch, bias=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512, out_features=out_ch, bias=True), )

        self.init_params()

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        prediction = self.classifier(feature)
        if self.bg_classifier == 'independence':
            prediction = prediction.view(-1, 2, 2)
        return {'feature': feature, 'prediction': prediction}


if __name__ == "__main__":
    input = torch.rand((32, 5, 224, 224))
    model = VGG11BN('none', False)
    output = model(input)
    print([x.shape for x in output.values()])
