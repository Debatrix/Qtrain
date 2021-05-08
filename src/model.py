import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from src.arch.module import Resnet18_encoder, AttUNet_decoder, Attention_pooling
from src.framework import Module


class AUnet(Module):
    def __init__(self,
                 mask_type='soft',
                 img_down=['bilinear', 4],
                 att_down='avgpool',
                 img_ch=1,
                 frozen=''):
        super(AUnet, self).__init__()
        img_down[0] = img_down[0].lower()
        att_down = att_down.lower()
        self.mask_type = mask_type
        seghead_ch = 3 if mask_type == 'hard' else 2

        if img_down[0] == 'conv':
            self.downsample = nn.Sequential(
                nn.Conv2d(img_ch, 64, 16, 8, 4, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif img_down[0] in ['nearest', 'bilinear', 'bicubic']:
            self.downsample = nn.Sequential(
                nn.Upsample(scale_factor=1 / img_down[1], mode=img_down[0]),
                nn.Conv2d(img_ch, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        else:
            raise ValueError('Unsupported input image downsample type: ' +
                             img_down)

        self.seghead = nn.Conv2d(64, seghead_ch, 1, 1, 0)
        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=2, bias=True),
        )

        self.init_params()

        self.encoder = Resnet18_encoder()
        self.decoder = AttUNet_decoder()

        self.attpooling = Attention_pooling(mask_type, att_down)

        need_frozen_list = []
        if frozen.lower() == 'body':
            need_frozen_list += [x for x in self.downsample.parameters()]
            need_frozen_list += [x for x in self.encoder.parameters()]
        elif frozen.lower() == 'seg':
            need_frozen_list += [x for x in self.downsample.parameters()]
            need_frozen_list += [x for x in self.encoder.parameters()]
            need_frozen_list += [x for x in self.decoder.parameters()]
            need_frozen_list += [x for x in self.seghead.parameters()]
        elif frozen.lower() == 'dfs':
            need_frozen_list += [x for x in self.linear.parameters()]
        if len(need_frozen_list) > 0:
            for param in need_frozen_list:
                param.requires_grad = False

    def forward(self, input):
        feature = [None, None, None, None, None]
        feature[0] = self.downsample(input)
        feature[1:] = self.encoder(feature[0])
        mask = self.seghead(self.decoder(*feature))
        output = self.attpooling(feature[-1], mask)
        output = self.linear(output)

        return output, mask


class BUnet(Module):
    def __init__(self,
                 mask_type='soft',
                 img_down=['bilinear', 4],
                 att_down='avgpool',
                 img_ch=1,
                 frozen=''):
        super(BUnet, self).__init__()
        img_down[0] = img_down[0].lower()
        att_down = att_down.lower()
        self.mask_type = mask_type
        seghead_ch = 3 if mask_type == 'hard' else 2

        if img_down[0] == 'conv':
            self.downsample = nn.Sequential(
                nn.Conv2d(img_ch, 64, 16, 8, 4, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif img_down[0] in ['nearest', 'bilinear', 'bicubic']:
            self.downsample = nn.Sequential(
                nn.Upsample(scale_factor=1 / img_down[1], mode=img_down[0]),
                nn.Conv2d(img_ch, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        else:
            raise ValueError('Unsupported input image downsample type: ' +
                             img_down)

        self.seghead = nn.Conv2d(64, seghead_ch, 1, 1, 0)
        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=2, bias=True),
        )

        self.init_params()

        self.encoder = Resnet18_encoder()
        self.decoder = AttUNet_decoder()

        self.attpooling = nn.AdaptiveAvgPool2d(1)

        need_frozen_list = []
        if frozen.lower() == 'body':
            need_frozen_list += [x for x in self.downsample.parameters()]
            need_frozen_list += [x for x in self.encoder.parameters()]
        elif frozen.lower() == 'seg':
            need_frozen_list += [x for x in self.downsample.parameters()]
            need_frozen_list += [x for x in self.encoder.parameters()]
            need_frozen_list += [x for x in self.decoder.parameters()]
            need_frozen_list += [x for x in self.seghead.parameters()]
        elif frozen.lower() == 'dfs':
            need_frozen_list += [x for x in self.linear.parameters()]
        if len(need_frozen_list) > 0:
            for param in need_frozen_list:
                param.requires_grad = False

    def forward(self, input):
        feature = [None, None, None, None, None]
        feature[0] = self.downsample(input)
        feature[1:] = self.encoder(feature[0])
        mask = self.seghead(self.decoder(*feature))
        output = self.attpooling(feature[-1]).view(feature[-1].shape[0], -1)
        output = self.linear(output)

        return output, mask


class CUnet(Module):
    def __init__(self,
                 mask_type='soft',
                 img_down=['bilinear', 4],
                 att_down='avgpool',
                 img_ch=1,
                 frozen=''):
        super(CUnet, self).__init__()
        img_down[0] = img_down[0].lower()

        if img_down[0] == 'conv':
            self.downsample = nn.Sequential(
                nn.Conv2d(img_ch, 64, 16, 8, 4, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        elif img_down[0] in ['nearest', 'bilinear', 'bicubic']:
            self.downsample = nn.Sequential(
                nn.Upsample(scale_factor=1 / img_down[1], mode=img_down[0]),
                nn.Conv2d(img_ch, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        else:
            raise ValueError('Unsupported input image downsample type: ' +
                             img_down)

        self.attpooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=2, bias=True),
        )

        self.init_params()

        self.encoder = Resnet18_encoder()

        need_frozen_list = []
        if frozen.lower() == 'body':
            need_frozen_list += [x for x in self.downsample.parameters()]
            need_frozen_list += [x for x in self.encoder.parameters()]
        elif frozen.lower() == 'dfs':
            need_frozen_list += [x for x in self.linear.parameters()]
        if len(need_frozen_list) > 0:
            for param in need_frozen_list:
                param.requires_grad = False

    def forward(self, input):
        feature = [None, None, None, None, None]
        feature[0] = self.downsample(input)
        feature[1:] = self.encoder(feature[0])
        mask = torch.zeros_like(feature[0])
        output = self.attpooling(feature[-1]).view(feature[-1].shape[0], -1)
        output = self.linear(output)

        return output, mask


if __name__ == "__main__":

    import cv2
    import time
    from tqdm import tqdm
    import numpy as np

    from torch.utils.data import DataLoader
    from src.dataset import FaceDataset

    # warmup
    model = CUnet().cuda(2)
    for _ in tqdm(range(5)):
        frame = np.random.random((1, 1, 1728, 2352))
        frame = torch.from_numpy(frame).to(torch.float32).cuda(2)
        with torch.no_grad():
            output, mask = model(frame)

    # result = '\n' + '#' * 40 + '\n'
    # for down1 in [
    #         # 'conv',
    #         'nearest',
    #         'bilinear',
    #         'bicubic',
    # ]:
    #     for down2 in [
    #             'maxpool',
    #             'avgpool',
    #             'nearest',
    #             'bilinear',
    #             'bicubic',
    #     ]:
    #         val_data = FaceDataset('CASIA-Iris-Distance',
    #                                mode='test',
    #                                less_data=0.005)
    #         val_data_loader = DataLoader(val_data, 1)
    #         model = AUnet(img_down=down1, att_down=down2).cuda(2)
    #         all_time = 0
    #         for data in tqdm(val_data_loader):
    #             start = time.time()
    #             frame = data['img'].cuda(2)
    #             with torch.no_grad():
    #                 output, mask = model(frame)
    #             all_time += time.time() - start
    #         result += '{} {}: {} fps\n'.format(down1, down2,
    #                                            len(val_data_loader) / all_time)
    # print(result + '#' * 40 + '\n')
