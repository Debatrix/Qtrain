import cv2
import json
import math
import random
import numpy as np
import os.path as osp

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as vtf


def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0]**2
    y_term = array_like_hm[:, 1]**2
    exp_value = -(x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)


def generate_heatmap(points, size):
    '''
    :param points:  [[x,y,sigma]]
    :return: heatmap
    '''
    x = np.arange(size[1], dtype=np.float)
    y = np.arange(size[0], dtype=np.float)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    heatmap = []

    for joint_id in range(len(points)):
        mu_x = int(points[joint_id][0])
        mu_y = int(points[joint_id][1])
        sigma = points[joint_id][2]
        zz = gaussian(xxyy.copy(), (mu_x, mu_y), sigma)
        heatmap.append(zz.reshape(size))
    heatmap = np.stack(heatmap, 0)

    return heatmap


# ##############################################################################


def transform(img, loc, p=0.3, offset=0.2, angle=45):
    if random.random() < p:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        if random.random() < 0.5:
            offset_h = np.random.randint(-int(offset * img.shape[-2]),
                                         int(offset * img.shape[-2]))
            offset_w = np.random.randint(-int(offset * img.shape[-1]),
                                         int(offset * img.shape[-1]))
            transMat = np.float32([[1, 0, offset_w], [0, 1, offset_h]])
            loc = loc + np.array((offset_w, offset_h, 0))
            img = cv2.warpAffine(img, transMat, (w, h))
        if random.random() < 0.5:
            angle = np.random.randint(-angle, angle)
            rotmat = cv2.getRotationMatrix2D(center, angle, 1)
            loc[0, :2] = np.dot(rotmat[:, :2], loc[0, :2]) + rotmat[:, 2]
            loc[1, :2] = np.dot(rotmat[:, :2], loc[1, :2]) + rotmat[:, 2]
            img = cv2.warpAffine(img, rotmat, (w, h))
    return img, loc


# ##############################################################################


class FaceDataset(data.Dataset):
    def __init__(self,
                 dataset='CASIA-Complex-CX3',
                 extractor='maxout',
                 mode='train',
                 less_data=False,
                 img_resize=False,
                 resize=8,
                 mask_type='soft',
                 enhance=0.3,
                 load_sim=False):
        super(FaceDataset, self).__init__()
        self.mode = mode
        self.extractor = extractor
        self.dataset_path = osp.join('dataset', dataset)

        self.enhance = enhance
        self.img_resize = img_resize
        if resize >= 1:
            self.resize = resize
        elif 0 < resize < 1:
            self.resize = 1 / resize
        else:
            self.resize = 1
        self.mask_type = mask_type

        meta_path = osp.join('dataset', dataset, 'meta_' + extractor + '.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.img_shape = meta['face_img_shape']

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
            self.info_list = self.info_list[:int(less_data *
                                                 len(self.info_list))]

        self.normalize = transforms.Normalize(*meta['face_mean&std'],
                                              inplace=True)

        if load_sim:
            print('Loading similarity')
            self.sim = torch.load(
                osp.join('dataset', dataset, 'feature',
                         'similarity_{}_{}.pth'.format(extractor, dataset)))
            self.sim['name'] = {
                osp.basename(x[0]).split('.')[0]: y
                for y, x in enumerate(self.sim['name'])
            }
            self.sim['label'] = {
                y: int(x[0])
                for y, x in enumerate(self.sim['label'])
            }
            pass
        else:
            self.sim = None

    def __len__(self) -> int:
        return len(self.info_list)

    def __getitem__(self, item):
        info = self.info_list[item]
        dfs = torch.tensor([info['l_dfs'], info['r_dfs']]).to(torch.float)
        name = [
            osp.basename(info['img']).split('.')[0],
            osp.basename(str(info['l_norm'])).split('.')[0],
            osp.basename(str(info['r_norm'])).split('.')[0],
        ]

        img = cv2.imread(osp.join(self.dataset_path, info['img']),
                         cv2.IMREAD_GRAYSCALE)

        loc = np.array([info['l_loc'], info['r_loc']])
        img, loc = transform(img, loc, self.enhance)

        lm = loc[:, 0] - loc[:, 2]
        rm = self.img_shape[1] - loc[:, 0] - loc[:, 2]
        um = loc[:, 1] - loc[:, 2]
        dm = self.img_shape[0] - loc[:, 1] - loc[:, 2]
        margin = np.stack((lm, rm, um, dm))
        if (margin[0, :] < 0.75 * loc[0, 2]).any():
            dfs[0] = 0
        if (margin[1, :] < 0.75 * loc[1, 2]).any():
            dfs[1] = 0

        loc = loc // self.resize
        if self.mask_type == 'soft':
            mask = generate_heatmap(
                loc, (self.img_shape[0] // self.resize,
                      self.img_shape[1] // self.resize)).astype(np.float32)
        else:
            mask = np.zeros((self.img_shape[0] // self.resize,
                             self.img_shape[1] // self.resize))
            mask = cv2.circle(mask, (loc[0, 0], loc[0, 1]),
                              loc[0, 2],
                              color=(1, 1, 1),
                              thickness=-1)
            mask = cv2.circle(mask, (loc[1, 0], loc[1, 1]),
                              loc[1, 2],
                              color=(2, 2, 2),
                              thickness=-1)
            mask = mask.astype(np.int)
        if self.img_resize:
            img = cv2.imresize(img, (self.img_shape[0] // self.resize,
                                     self.img_shape[1] // self.resize))

        img = torch.tensor(img / 255).unsqueeze(0).to(torch.float)
        img = self.normalize(img)
        mask = torch.tensor(mask)
        loc = torch.tensor(loc).to(torch.float)

        return {'img': img, 'mask': mask, 'dfs': dfs, 'name': name, 'loc': loc}


if __name__ == "__main__":
    from tqdm import tqdm
    from itertools import cycle
    dataset1 = FaceDataset('CASIA-Iris-Distance',
                           mode='val',
                           less_data=1,
                           enhance=1,
                           mask_type='soft',
                           load_sim=False)
    dataloader1 = data.DataLoader(dataset1, 10, shuffle=True, drop_last=True)
    dataset2 = FaceDataset('CASIA-Complex-CX3',
                           mode='val',
                           less_data=1,
                           enhance=1,
                           mask_type='soft',
                           load_sim=False)
    dataloader2 = data.DataLoader(dataset2, 10, shuffle=True, drop_last=True)
    for x in zip(cycle(dataloader1), dataloader2):
        print(x)
    # for x in tqdm(dataloader):
    #     print(x['img'].shape, x['mask'].shape, x['dfs'], x['name'],
    #           x['loc'].shape, x['loc'])
