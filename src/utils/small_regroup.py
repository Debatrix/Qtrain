# %% [markdown]
# #医学图像数据重新整理
#
# %%
import cv2
import os
import json
import time
import os.path as osp
from tqdm import tqdm
from glob import glob
import numpy as np
from libtiff import TIFF
import matplotlib.pyplot as plt

# %%
dir_path = [
    '/home/dl/dataset/省医乳腺-组织块/可疑组织', '/home/dl/dataset/省医乳腺-组织块/正常组织',
    '/home/dl/dataset/省医乳腺-组织块/肿瘤组织'
]
sub_path = ['ctr0', 'ctr1', 'ctr2', 'ctr3', 'ctr4']
dst_path = 'dataset/BreastCancer/data'
json_path = 'dataset/BreastCancer/meta/raw_meta.json'


# %%
def load_tif(path):
    assert os.path.exists(path)
    image = []
    tif = TIFF.open(path, 'r')
    for img in tif.iter_images():
        image.append(img)
    return image


# %%
def save_preview(src, dst):
    data = np.load(src)
    img = np.stack((data[0], data[1], data[4]), -1)
    img += data[2:4].sum(0)[:, :, np.newaxis]
    img = ((img - img.min()) / (img.max() - img.min())).round().astype(
        np.uint8)
    cv2.imwrite(dst, img)


# %%
big_list = []
for p in dir_path:
    big_list += glob(os.path.join(p, '*'))

# %%
meta = {}
num = -1
all_start = time.time()
for big in tqdm(big_list):
    num += 1
    print('{}/{}:\t{}'.format(num, len(big_list), big))
    label, big_name = osp.split(big)
    big_name = big_name.split(' ')[0]
    label = osp.basename(label)
    # label = 'unlabel' if label not in ['tumour', 'normal'] else label
    if label in ['normal', '正常组织']:
        label = 'normal'
    elif label in ['tumour', '肿瘤组织']:
        label = 'tumour'
    else:
        label = 'unlabel'
    print('Loading image: {}\t{}'.format(big_name, label))
    big_img = []
    try:
        start = time.time()
        for x in sub_path:
            path = os.path.join(big, 'dev1', x, 'add_z001.tif')
            big_img.append(load_tif(path))
        assert len(set(len(x) for x in big_img)) == 1
        print('Loading image spend: {}s\n Saving .npy...'.format(
            int(time.time() - start)))

        for idx in range(len(big_img[0])):
            sub_name = '{}_{:0>4d}'.format(big_name, idx)
            small_img = [x[idx] for x in big_img]
            small_img = np.stack(small_img, 0)

            meta[sub_name] = {
                'file': sub_name + '.npy',
                'instance': big_name,
                'index': idx,
                'class': label,
                'sub_class': None,
                'mean': small_img.mean((1, 2)).tolist(),
            }
            np.save(osp.join(dst_path, sub_name + '.npy'), small_img)
    except Exception as e:
        print(e)
print('{:.2f}min in total'.format((time.time() - all_start) / 60))

with open(json_path, 'w') as f:
    json.dump({'label': meta}, f)
