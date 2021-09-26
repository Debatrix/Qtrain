import cv2
import json
import math
import random
import numpy as np
import os.path as osp
from PIL import Image
from itertools import permutations

import torch
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as transforms


# ##############################################################################
class FaceDataset(data.Dataset):
    def __init__(self,
                 dataset='distance',
                 mode='qtrain',
                 less_data=False,
                 dfs={},
                 **kwargs):
        super(FaceDataset, self).__init__()

        self.mode = mode

        self.dataset_path = osp.join('dataset', dataset)

        meta_path = osp.join(self.dataset_path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        if meta['name'] == 'CASIA-Iris-Distance':
            self.rate = 4.59375
        else:
            self.rate = 2.5

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

        self.dfs = dfs if dfs is not None else {}

    def __len__(self) -> int:
        return len(self.info_list)

    def _load_img(self, info):
        img = Image.open(osp.join(self.dataset_path, info['img']))
        img = self.transform(img)
        img = self.normalize(img)
        l_loc = np.round(np.array(info['l_loc']) / self.rate).astype(np.int)
        r_loc = np.round(np.array(info['r_loc']) / self.rate).astype(np.int)

        # image are resized!
        mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
        mask = cv2.circle(mask, (l_loc[0], l_loc[1]), l_loc[2], (1, 1, 1), -1)
        mask = cv2.circle(mask, (r_loc[0], r_loc[1]), r_loc[2], (2, 2, 2), -1)
        mask = torch.from_numpy(mask)

        l_name = osp.basename(info['l_norm']).split('.')[0]
        r_name = osp.basename(info['r_norm']).split('.')[0]
        ename = [l_name, r_name]

        l_dfs = self.dfs[l_name] if l_name in self.dfs else 0
        r_dfs = self.dfs[r_name] if r_name in self.dfs else 0
        dfs = torch.tensor((l_dfs, r_dfs), dtype=torch.float)

        return img.to(torch.float), mask.to(torch.long), dfs, ename

    def __getitem__(self, item):
        info = self.info_list[item]
        img, mask, dfs, ename = self._load_img(info)
        return {'img': img, 'mask': mask, 'dfs': dfs, 'ename': ename}


class EyeDataset(data.Dataset):
    def __init__(self,
                 dataset='distance',
                 mode='rtrain',
                 less_data=False,
                 dfs=None,
                 weight='gaussian',
                 label_offset=0,
                 **kwargs):
        super(EyeDataset, self).__init__()

        self.mode = mode
        self.weight = weight
        self.label_offset = label_offset

        self.dataset_path = osp.join('dataset', dataset)

        meta_path = osp.join(self.dataset_path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.meta = meta

        # # delete this!
        if mode == 'rtrain':
            meta['protocol'][
                mode] = meta['protocol'][mode] + meta['protocol']['qtrain']

        self.num_classes = meta['iris_label_num']
        self.info_list = []
        for k, info in meta['info'].items():
            if info['label'] in meta['protocol'][mode]:
                name = osp.basename(info['l_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'face': k,
                    'norm': info['l_norm'],
                    'label': meta['iris_label'][name]
                }
                if osp.exists(osp.join(self.dataset_path, einfo['norm'])):
                    self.info_list.append(einfo)
                name = osp.basename(info['r_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'face': k,
                    'norm': info['r_norm'],
                    'label': meta['iris_label'][name]
                }
                if osp.exists(osp.join(self.dataset_path, einfo['norm'])):
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

        if dfs is not None and weight is not None:
            self.dfs = dfs
            dfs = np.array([x for x in dfs.values()])
            if weight == 'gaussian':
                self.weight_fun = lambda x: ((1 / (np.power(
                    2 * np.pi, 0.5) * dfs.std())) * np.exp(-0.5 * np.power(
                        (x - dfs.mean()) / dfs.std(), 2)))
            else:
                self.weight_fun = None
        else:
            self.dfs = {}

    def __len__(self) -> int:
        return len(self.info_list)

    def set_label_offset(self, offset):
        self.label_offset = offset
        return self.meta['iris_label_num'] + offset

    def _load_img(self, info):
        img = Image.open(osp.join(self.dataset_path, info['norm']))
        img = self.transform(img)

        label = torch.tensor(info['label'] + self.label_offset,
                             dtype=torch.long)
        name = info['name']

        if name in self.dfs and self.weight is not None:
            weight = torch.tensor(self.weight_fun(self.dfs[name]),
                                  dtype=torch.float)
        else:
            weight = torch.tensor(1, dtype=torch.float)

        return img.to(torch.float), label, weight, name

    def __getitem__(self, item):
        info = self.info_list[item]
        img, label, weight, name = self._load_img(info)
        return {'img': img, 'name': name, 'label': label, 'weight': weight}


class EyePairDataset(data.Dataset):
    def __init__(self,
                 dataset='distance',
                 mode='rtrain',
                 less_data=False,
                 dfs=None,
                 weight='gaussian',
                 **kwargs):
        super(EyePairDataset, self).__init__()

        self.mode = mode
        self.weight = weight

        self.dataset_path = osp.join('dataset', dataset)

        meta_path = osp.join(self.dataset_path, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # delete this!
        if mode == 'test':
            meta['protocol'][
                mode] = meta['protocol'][mode] + meta['protocol']['qtrain']

        self.num_classes = meta['iris_label_num']
        self.info_list = []
        info_list = []
        label_dict = {x: [] for x in meta['iris_label'].values()}
        for k, info in meta['info'].items():
            if info['label'] in meta['protocol'][mode]:
                name = osp.basename(info['l_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'face': k,
                    'norm': info['l_norm'],
                    'label': meta['iris_label'][name]
                }
                if osp.exists(osp.join(self.dataset_path, einfo['norm'])):
                    label_dict[meta['iris_label'][name]].append(einfo)
                    info_list.append(einfo)
                name = osp.basename(info['r_norm']).split('.')[0]
                einfo = {
                    'name': name,
                    'face': k,
                    'norm': info['r_norm'],
                    'label': meta['iris_label'][name]
                }
                if osp.exists(osp.join(self.dataset_path, einfo['norm'])):
                    label_dict[meta['iris_label'][name]].append(einfo)
                    info_list.append(einfo)
        for k, v in label_dict.items():
            if len(v) == 0:
                continue
            _pos = [x for x in permutations(v, 2)]
            _neg = []
            while len(_neg) <= len(_pos) * 2:
                neg = np.random.choice(info_list)
                if neg['label'] != k:
                    _neg.append(neg)
            for idx, pos in enumerate(_pos):
                self.info_list.append(list(pos) + [_neg[idx]])

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

        return img.to(torch.float)

    def __getitem__(self, item):
        info = self.info_list[item]
        img1 = self._load_img(info[0])
        img2 = self._load_img(info[1])
        img3 = self._load_img(info[1])
        assert info[0]['label'] == info[1]['label']
        assert info[0]['label'] != info[2]['label']
        return {
            'img1': img1,
            'img2': img2,
            'img3': img3,
        }


# #############################################################################
def get_eye_dataset(datasets, **kwargs):
    if isinstance(datasets, str):
        datasets = [datasets]

    assert isinstance(datasets, (list, tuple, np.ndarray))

    offset = 0
    dataset_list = []
    for dataset in datasets:
        dataset_list.append(EyeDataset(dataset=dataset, **kwargs))
        offset = dataset_list[-1].set_label_offset(offset)

    if len(dataset_list) == 1:
        return dataset_list[0], offset
    else:
        return ConcatDataset(dataset_list), offset


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    # data = EyeDataset(mode='rtrain', dataset='LG4000', less_data=0.1)
    data, num_class = get_eye_dataset(
        ['LG4000', 'LG2200', 'distance', 'thousand'], mode='rtrain')
    print(num_class, len(data))
    data = DataLoader(
        data,
        32,
        shuffle=True,
        drop_last=True,
    )
    print(len(data))

    for x in data:
        # print(x['img'].shape, x['mask'].shape, x['ename'], x['dfs'])
        # print(x['img1'].shape, x['img2'].shape, x['img3'].shape)
        print(x['img'].shape, x['name'], x['label'])
