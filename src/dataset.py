import cv2
import json
import math
import random
import numpy as np
import os.path as osp
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


# ##############################################################################
class FaceDataset(data.Dataset):
    def __init__(self,
                 dataset='distance',
                 mode='train',
                 less_data=False,
                 **kwargs):
        super(FaceDataset, self).__init__()

        self.mode = mode

        self.dataset_path = osp.join('dataset', dataset)

        meta_path = osp.join(self.dataset_path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.info_list = []
        for info in meta['info'].values():
            if info['label'] in meta['protocol'][mode]:
                self.info_list.append(info)
        random.shuffle(self.info_list)

        if less_data:
            if isinstance(less_data, (float, int)):
                if less_data > 1 or less_data <= 0:
                    less_data = 0.05
            else:
                less_data = 0.05
            num = int(less_data * len(self.info_list))
            num = num if num > 128 else 128
            self.info_list = self.info_list[:num]

        self.transform = transforms.Compose([
            # transforms.Resize(size=[128, 128]),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(*meta['face_mean&std'],
                                              inplace=True)

    def __len__(self) -> int:
        return len(self.info_list)

    def _load_img(self, info):
        img = Image.open(osp.join(self.dataset_path, info['img']))
        img = self.transform(img)
        img = self.normalize(img)

        loc = torch.tensor([info['l_loc'], info['r_loc']])
        mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
        mask = cv2.circle(mask, (info['l_loc'][0], info['l_loc'][1]),
                          info['l_loc'][2], (1, 1, 1), -1)
        mask = cv2.circle(mask, (info['r_loc'][0], info['r_loc'][1]),
                          info['r_loc'][2], (2, 2, 2), -1)
        mask = torch.from_numpy(mask)

        return img.to(torch.float), mask.to(torch.long), loc.to(torch.long)

    def __getitem__(self, item):
        info = self.info_list[item]
        img, mask, loc = self._load_img(info)
        return {'img': img, 'mask': mask, 'loc': loc}


class EyeDataset(data.Dataset):
    def __init__(self,
                 dataset='distance',
                 mode='train',
                 less_data=False,
                 **kwargs):
        super(EyeDataset, self).__init__()

        self.mode = mode

        self.dataset_path = osp.join('dataset', dataset)

        meta_path = osp.join(self.dataset_path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.info_list = []
        for info in meta['info'].values():
            if info['label'] in meta['protocol'][mode]:
                name = osp.basename(info['l_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'norm': info['l_norm'],
                    'label': meta['iris_label'][name]
                }
                self.info_list.append(einfo)
                name = osp.basename(info['r_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'norm': info['r_norm'],
                    'label': meta['iris_label'][name]
                }
                self.info_list.append(einfo)
        random.shuffle(self.info_list)

        if less_data:
            if isinstance(less_data, (float, int)):
                if less_data > 1 or less_data <= 0:
                    less_data = 0.05
            else:
                less_data = 0.05
            num = int(less_data * len(self.info_list))
            num = num if num > 128 else 128
            self.info_list = self.info_list[:num]

        self.transform = transforms.Compose([
            transforms.Resize(size=[128, 128]),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.info_list)

    def _load_img(self, info):
        img = Image.open(osp.join(self.dataset_path, info['norm']))
        img = self.transform(img)

        label = torch.tensor(info['label'])
        name = info['name']

        return img.to(torch.float), label.to(torch.long), name

    def __getitem__(self, item):
        info = self.info_list[item]
        img, label, name = self._load_img(info)
        return {'img': img, 'name': name, 'label': label}


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    data = EyeDataset(mode='val', dataset='distance', less_data=0.1)

    for x in data:
        try:
            # print(x['img'].shape, x['name'], x['label'])
            pass
        except Exception as e:
            print(e)
