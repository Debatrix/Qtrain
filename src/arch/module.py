import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torchvision
from torchvision.models import ResNet

from src.framework import Module


# #############################################################################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 *,
                 reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 *,
                 reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# ############################################################################
class mfm_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(mfm_conv, self).__init__()
        self.out_channels = out_channels
        self.filter = nn.Conv2d(in_channels,
                                2 * out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class mfm_linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(mfm_linear, self).__init__()
        self.out_channels = out_channels
        self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class MaxoutBackbone(Module):
    def __init__(self):
        super(MaxoutBackbone, self).__init__()
        self.features = nn.Sequential(
            mfm_conv(1, 48, 9, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm_conv(48, 96, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm_conv(96, 128, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm_conv(128, 192, 4, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm_linear(5 * 5 * 192, 256)
        self.dropout = nn.Dropout(0.7)

        self.init_params()

    def forward(self, x, evaluation=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        if evaluation:
            x = F.normalize(x, dim=1)
        return x


# #############################################################################


class VGG11BNBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG11BNBackbone, self).__init__()

        model = torchvision.models.vgg11_bn(pretrained)

        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, 512)
        return feature


class Resnet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18Backbone, self).__init__()

        model = torchvision.models.resnet18(pretrained)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        return feature


class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetBackbone, self).__init__()

        model = torchvision.models.mobilenet_v2(pretrained)

        self.features = model.features
        self.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        init.kaiming_normal_(self.features[0].weight, a=0, mode='fan_in')

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        return feature


class SEnet18(Module):
    def __init__(self):
        super(SEnet18, self).__init__()
        model = se_resnet18(256)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.init_params()

    def forward(self, input):
        feature = self.features(input)
        feature = self.avgpool(feature)
        feature = feature.view(-1, feature.shape[1])
        return feature


# #############################################################################


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class Attention_pooling(nn.Module):
    def __init__(self, mask_type='soft', down_type='bilinear'):
        super(Attention_pooling, self).__init__()
        self.mask_type = mask_type.lower()
        self.down_type = down_type.lower()

        self.eps = 1e-12

    def forward(self, feature, mask):
        fh, fw = feature.shape[-2:]
        mh, mw = mask.shape[-2:]
        if fh > mh and fw > mw:
            if self.down_type in ['nearest', 'bilinear', 'bicubic']:
                feature = F.interpolate(feature, (mh, mw), mode=self.down_type)
            elif self.down_type == 'maxpool':
                feature = F.adaptive_max_pool2d(feature, (mh, mw))
            elif self.down_type == 'avgpool':
                feature = F.adaptive_avg_pool2d(feature, (mh, mw))
            else:
                raise ValueError(
                    'Unsupported Attention_pooling downsample type: ' +
                    self.down_type)
        elif mh > fh and mw > fw:
            if self.down_type in ['nearest', 'bilinear', 'bicubic']:
                mask = F.interpolate(mask, (fh, fw), mode=self.down_type)
            elif self.down_type == 'maxpool':
                mask = F.adaptive_max_pool2d(mask, (fh, fw))
            elif self.down_type == 'avgpool':
                mask = F.adaptive_avg_pool2d(mask, (fh, fw))
            else:
                raise ValueError(
                    'Unsupported Attention_pooling downsample type: ' +
                    self.down_type)
        elif mh == fh and mw == fw:
            pass
        else:
            raise ValueError('Mask size {}, feature size {}'.format(
                mask.shape, feature.shape))

        if self.mask_type == 'hard':
            mask = torch.softmax(mask, 1)[:, 1:, :, :].sum(1).unsqueeze(1)
        else:
            mask = torch.sigmoid(mask[:, 1:, :, :]).sum(1).unsqueeze(1)

        output = torch.sum(feature * mask, dim=(2, 3))
        output = output / (torch.sum(mask, dim=(2, 3)) + self.eps)
        output = output.view(output.shape[0], -1)
        return output


class Resnet18_encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18_encoder, self).__init__()
        model = torchvision.models.resnet18(pretrained)

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, 16, 8, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        init.kaiming_normal_(self.layer0[0].weight, a=0, mode='fan_in')
        init.constant_(self.layer0[1].weight, 1)
        init.constant_(self.layer0[1].bias.data, 0.0)

        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, input):
        out1 = self.layer0(input)
        out2 = self.layer1(self.maxpool(out1))
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5


class UNet_encoder(Module):
    def __init__(self, channel=16):
        super(UNet_encoder, self).__init__()
        filters = [channel, channel * 2, channel * 4]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(1, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.init_params()

    def forward(self, input):

        out1 = self.Conv1(input)

        out2 = self.Maxpool1(out1)
        out2 = self.Conv2(out2)

        out3 = self.Maxpool2(out2)
        out3 = self.Conv3(out3)
        return out1, out2, out3


class UNet_decoder(Module):
    def __init__(self, channel=16, seghead_ch=3):
        super(UNet_decoder, self).__init__()

        filters = [channel, channel * 2, channel * 4]
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], seghead_ch, 1, 1, 0)

        self.init_params()

    def forward(self, out1, out2, out3):
        dout3 = self.Up3(out3)
        dout3 = F.interpolate(dout3, (out2.shape[2], out2.shape[3]),
                              mode='bilinear',
                              align_corners=True)
        dout3 = torch.cat((out2, dout3), dim=1)
        dout3 = self.Up_conv3(dout3)

        dout2 = self.Up2(dout3)
        dout2 = F.interpolate(dout2, (out1.shape[2], out1.shape[3]),
                              mode='bilinear',
                              align_corners=True)
        dout2 = torch.cat((out1, dout2), dim=1)
        dout2 = self.Up_conv2(dout2)

        dout = self.Conv(dout2)

        return dout


class AttUNet_encoder(Module):
    def __init__(self, channel=64):
        super(AttUNet_encoder, self).__init__()
        ch = [channel * 2**x for x in range(5)]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=1, out_ch=ch[0])
        self.Conv2 = conv_block(in_ch=ch[0], out_ch=ch[1])
        self.Conv3 = conv_block(in_ch=ch[1], out_ch=ch[2])
        self.Conv4 = conv_block(in_ch=ch[2], out_ch=ch[3])
        self.Conv5 = conv_block(in_ch=ch[3], out_ch=ch[4])

        self.init_params()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return x1, x2, x3, x4, x5


class AttUNet_decoder(Module):
    def __init__(self, seghead_ch=3, channels=[64, 128, 256, 512]):
        super(AttUNet_decoder, self).__init__()

        self.Up5 = up_conv(in_ch=channels[-1], out_ch=channels[-2])
        self.Att5 = Attention_block(F_g=channels[-2],
                                    F_l=channels[-2],
                                    F_int=channels[-2] // 2)
        self.Up_conv5 = conv_block(in_ch=channels[-1], out_ch=channels[-2])

        self.Up4 = up_conv(in_ch=channels[-2], out_ch=channels[-3])
        self.Att4 = Attention_block(F_g=channels[-3],
                                    F_l=channels[-3],
                                    F_int=channels[-3] // 2)
        self.Up_conv4 = conv_block(in_ch=channels[-2], out_ch=channels[-3])

        self.Up3 = up_conv(in_ch=channels[-3], out_ch=channels[-4])
        self.Att3 = Attention_block(F_g=channels[-4],
                                    F_l=channels[-4],
                                    F_int=channels[-4] // 2)
        self.Up_conv3 = conv_block(in_ch=channels[-3], out_ch=channels[-4])

        self.Up2 = up_conv(in_ch=channels[-4], out_ch=channels[-4])
        self.Att2 = Attention_block(F_g=channels[-4],
                                    F_l=channels[-4],
                                    F_int=channels[-4] // 2)
        self.Up_conv2 = conv_block(in_ch=channels[-3], out_ch=channels[-4])
        self.Conv = nn.Conv2d(channels[-4], seghead_ch, 1, 1, 0)

        self.init_params()

    def forward(self, x1, x2, x3, x4, x5):
        # decoding + concat path
        x5 = F.interpolate(x5, (x4.shape[2], x4.shape[3]),
                           mode='bilinear',
                           align_corners=True)
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d5 = F.interpolate(d5, (x3.shape[2], x3.shape[3]),
                           mode='bilinear',
                           align_corners=True)
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d4 = F.interpolate(d4, (x2.shape[2], x2.shape[3]),
                           mode='bilinear',
                           align_corners=True)
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d3 = F.interpolate(d3, (x1.shape[2], x1.shape[3]),
                           mode='bilinear',
                           align_corners=True)
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)

        return d1


# #############################################################################
class PredictHead(Module):
    def __init__(self, num_classes, feature_channel=256):
        super(PredictHead, self).__init__()
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(feature_channel, num_classes)

        self.init_params()

    def forward(self, x):
        x = self.fc(self.dropout(x))
        return x