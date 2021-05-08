import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision

from src.framework import Module


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

        output = torch.sum(feature * mask, dim=(2, 3)) / (
            torch.sum(mask, dim=(2, 3)) + self.eps)
        output = output.view(output.shape[0], -1)
        return output


class Resnet18_encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet18_encoder, self).__init__()
        model = torchvision.models.resnet18(pretrained)

        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, input):
        out2 = self.layer1(self.maxpool(input))
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5


class UNet_encoder(nn.Module):
    def __init__(self, channel=32):
        super(UNet_encoder, self).__init__()
        filters = [channel, channel * 2, channel * 4]
        in_ch = 1

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

    def forward(self, input):

        out1 = self.Conv1(input)

        out2 = self.Maxpool1(out1)
        out2 = self.Conv2(out2)

        out3 = self.Maxpool2(out2)
        out3 = self.Conv3(out3)
        return out1, out2, out3


class UNet_decoder(nn.Module):
    def __init__(self, channel=32):
        super(UNet_decoder, self).__init__()

        filters = [channel, channel * 2, channel * 4]
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], 2, 1, 1, 0)

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


class AttUNet_encoder(nn.Module):
    def __init__(self, channel=32):
        super(AttUNet_encoder, self).__init__()
        ch = [channel * 2**x for x in range(5)]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=channel, out_ch=ch[0])
        self.Conv2 = conv_block(in_ch=ch[0], out_ch=ch[1])
        self.Conv3 = conv_block(in_ch=ch[1], out_ch=ch[2])
        self.Conv4 = conv_block(in_ch=ch[2], out_ch=ch[3])
        self.Conv5 = conv_block(in_ch=ch[3], out_ch=ch[4])

        self.sigmoid = nn.Sigmoid()

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
    def __init__(self, out_ch=2, channels=[64, 128, 256, 512]):
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

        # d1 = self.Conv_1x1(d2)
        # d1 = self.sigmoid(d1)

        return d2


if __name__ == "__main__":
    import time
    t = 0
    encoder = Resnet18_encoder().cuda()
    decoder = AttUNet_decoder().cuda()
    with torch.no_grad():
        for _ in range(10):
            t1 = time.time()
            input = torch.rand((1, 1, 3072, 4080)).cuda()
            feature = encoder(input)
            mask = decoder(*feature)
            t += time.time() - t1
            print('feature.shape:', [x.shape for x in feature])
            print('mask.shape:', mask.shape)
    print((t / 10) * 1000)
