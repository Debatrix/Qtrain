# %% [markdown]
# 处理的数据完成数据集划分

# %%
import json
import random
import numpy as np
import os.path as osp
from tqdm import tqdm

# %%
src_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/meta/raw_meta.json'
dst_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/meta/2104.json'

img_dir_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/clean'

old_path = ''

split = np.array([6, 1, 3])

cal_mean_std = True
choice_rate = 0.01
thr = 9


# %%
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f)


# %%
# 划分数据集
protocol = [[], [], []]

data = load_json(src_path)
instance_list = set(x['instance'] for x in data['label'].values()
                    if x['class'] != 'unlabel')
if osp.exists(old_path):
    _protocol = load_json(old_path)['protocol']
    protocol[0] = list(set(_protocol[0]) & instance_list)
    protocol[1] = list(set(_protocol[1]) & instance_list)
    protocol[2] = list(set(_protocol[2]) & instance_list)
    _instance_list = set(protocol[0] + protocol[1] + protocol[2])
    instance_list = instance_list - _instance_list
instance_list = list(instance_list)
random.shuffle(instance_list)

s1 = int(np.round(len(instance_list) * split[0] / split.sum()))
s2 = int(s1 + np.round(len(instance_list) * split[1] / split.sum()))
protocol[0] += instance_list[:s1]
protocol[1] += instance_list[s1:s2]
protocol[2] += instance_list[s2:]
data['protocol'] = protocol
save_json(dst_path, data)

# %%
# 统计
print('train_slide:{}, val_slide:{}, test_slide:{}'.format(
    len(protocol[0]), len(protocol[1]), len(protocol[2])))

tile_list = [[], [], []]
for k, v in data['label'].items():
    if v['instance'] in protocol[0]:
        tile_list[0].append(k)
    if v['instance'] in protocol[1]:
        tile_list[1].append(k)
    if v['instance'] in protocol[2]:
        tile_list[2].append(k)
print('train_tile:{}, val_tile:{}, test_tile:{}'.format(
    len(tile_list[0]), len(tile_list[1]), len(tile_list[2])))

# %%
if cal_mean_std:
    path_list = [osp.join(img_dir_path, x + '.npy') for x in tile_list[0]]
    path_list = np.random.choice(path_list, int(choice_rate * len(path_list)))

    imgs = []
    labels = []
    for x in tqdm(path_list):
        img = np.load(x)
        imgs.append(img)
        if img.mean((-1, -2)).mean() < thr:
            labels.append(False)
        else:
            labels.append(True)
    imgs = np.stack(imgs, 0)
    labels = np.array(labels)
    param = [imgs.mean((0, -1, -2)).tolist(), imgs.std((0, -1, -2)).tolist()]
    param1 = [
        imgs[labels].mean((0, -1, -2)).tolist(), imgs[labels].std(
            (0, -1, -2)).tolist()
    ]
    if np.logical_not(labels).sum() < 1:
        param0 = [[0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1]]
    else:
        param0 = [
            imgs[np.logical_not(labels)].mean((0, -1, -2)).tolist(),
            imgs[np.logical_not(labels)].std((0, -1, -2)).tolist()
        ]
    print(param, '\n', param0, '\n', param1)

    data['mean_std'] = [param, param0, param1]
    save_json(dst_path, data)

# %%