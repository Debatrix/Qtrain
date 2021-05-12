# %% [markdown]
# # 医学图像数据重新整理
# 大图 带roi
# %%
import cv2
import os
import json
import time
import pandas as pd
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
from libtiff import TIFF
import matplotlib.pyplot as plt

# %%
dir_path = [
    '/home/dl/dataset/省医乳腺-组织块414/可疑组织', '/home/dl/dataset/省医乳腺-组织块414/正常组织',
    '/home/dl/dataset/省医乳腺-组织块414/肿瘤组织'
]

dst_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/clean'
json_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/meta/raw_meta.json'

tile_size = (650, 650)
mask_thr = 0.75

file_name_list = [
    'img-ctr0-0.tif',
    'img-ctr1-0.tif',
    'img-ctr2-0.tif',
    'img-ctr3-0.tif',
    'img-ctr4-0.tif',
]


# %%
def load_tif(path):
    assert os.path.exists(path)
    image = []
    tif = TIFF.open(path, 'r')
    for img in tif.iter_images():
        image.append(img)
    if len(image) == 1:
        image = image[0]
    return image


# %%
def load_xlsx(path):
    data = pd.read_excel(path, sheet_name=0)
    return data.values


# %%
def save_preview(src, dst):
    data = np.load(src)
    img = np.stack((data[0], data[1], data[4]), -1)
    img += data[2:4].sum(0)[:, :, np.newaxis]
    img = ((img - img.min()) / (img.max() - img.min())).round().astype(
        np.uint8)
    cv2.imwrite(dst, img)


# %%
# 遍历大图
big_list = []
for p in dir_path:
    big_list += glob(os.path.join(p, '*'))

# %%
# 读取文件夹中图像
log = []
meta = {}
slide_count = {'normal': 0, 'tumour': 0, 'unlabel': 0}
tile_count = {'normal': 0, 'tumour': 0, 'unlabel': 0}
num = -1
slide_num = 0
all_start = time.time()
for big in big_list:
    num += 1
    _log = '\n{}/{}:\t{}'.format(num, len(big_list), big)
    print(_log)
    log.append(_log)

    # label规范化
    label, big_name = osp.split(big)
    big_name = big_name.split(' ')[0]
    label = osp.basename(label)
    if label in ['normal', '正常组织']:
        label = 'normal'
    elif label in ['tumour', '肿瘤组织']:
        label = 'tumour'
    else:
        label = 'unlabel'

    # 载入图像
    _log = 'Loading image: {}\t{}'.format(big_name, label)
    print(_log)
    log.append(_log)
    try:
        start = time.time()
        big_img = []
        for x in file_name_list:
            path = os.path.join(big, 'dev1', x)
            assert osp.exists(path), 'error: tif not found!'
            big_img.append(load_tif(path))
        big_img = np.stack(big_img, 0)
        if big_img.max() > 255:
            big_img = np.clip(big_img, 0, 255)
            _log = 'warning: max bigger than 255!'
            print(_log)
            log.append(_log)
        big_img = big_img.astype(np.uint8)
    except Exception as e:
        print(e)
        log.append(_log)
        continue
    _log = 'Loading image spend: {}s'.format(int(time.time() - start))
    print(_log)
    log.append(_log)

    # 计数
    slide_num += 1
    slide_count[label] += 1

    # 载入roi
    img_size = big_img.shape[1:]
    mask_size = [
        int(np.ceil(img_size[0] / (tile_size[0] // 2))),
        int(np.ceil(img_size[1] / (tile_size[1] // 2)))
    ]
    mask = np.zeros(mask_size)

    roi_paths = glob(os.path.join(big, 'dev1', 'ROI*.xlsx'))

    if len(roi_paths) > 0:
        roi = []
        for x in roi_paths:
            r = load_xlsx(x)
            r[:, 0] = r[:, 0] / (tile_size[0] // 2)
            r[:, 1] = r[:, 1] / (tile_size[1] // 2)
            r = np.round(r).astype(np.int)
            roi.append(r)
        mask = cv2.fillPoly(mask, roi, (255, 255, 255))
    else:
        mask = np.ones(mask_size)
        _log = 'warning: no ROI*.xlsx found!'
        print(_log)
        log.append(_log)

    bg_paths = glob(os.path.join(big, 'dev1', 'BG*.xlsx'))
    if len(roi_paths) > 0:
        bg = []
        for x in bg_paths:
            r = load_xlsx(x)
            r[:, 0] = r[:, 0] / (tile_size[0] // 2)
            r[:, 1] = r[:, 1] / (tile_size[1] // 2)
            r = np.round(r).astype(np.int)
            bg.append(r)
        mask = cv2.fillPoly(mask, bg, (0, 0, 0))

    mask = mask.astype(np.bool)

    # 保存小图和标签
    pbar = tqdm(total=mask_size[0] * mask_size[1])
    for y in range(mask_size[0]):
        for x in range(mask_size[1]):
            pbar.update(1)
            if mask[y:y + 2, x:x + 2].sum() < 3:
                continue
            w_s = (tile_size[0] // 2) * x
            w_e = (tile_size[0] // 2) * (x + 2)
            h_s = (tile_size[0] // 2) * y
            h_e = (tile_size[0] // 2) * (y + 2)
            if w_e > img_size[1]:
                w_s, w_e = img_size[1] - tile_size[1], img_size[1]
            if h_e > img_size[0]:
                h_s, h_e = img_size[0] - tile_size[0], img_size[0]
            sub_name = '{}_{:0>5d}_{:0>5d}'.format(big_name, w_s, h_s)
            small_img = big_img[:, h_s:h_e, w_s:w_e]
            if small_img.shape[1:] != tile_size:
                _log = '{} tile size {}'.format(sub_name, small_img.shape)
                print(_log)
                log.append(_log)
            meta[sub_name] = {
                'instance': big_name,
                'index': (w_s, h_s),
                'class': label,
                'sub_class': None,
            }
            tile_count[label] += 1
            np.save(osp.join(dst_path, sub_name + '.npy'), small_img)
    pbar.close()

_log = '{:.2f}min in total'.format((time.time() - all_start) / 60)
print(_log)
log.append(_log)

with open(json_path, 'w') as f:
    json.dump({
        'label': meta,
        'tile_size': tile_size[0],
    }, f)

with open(osp.join(osp.dirname(json_path), 'regroup.log'), 'w') as f:
    for x in log:
        f.write(x + '\n')

print('slide num:', slide_num, slide_count)
print('tile num:', len(meta), tile_count)
# %%
