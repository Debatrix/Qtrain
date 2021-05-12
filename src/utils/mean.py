# %%
import json
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
# %%
c_rate = 0.025
bg_ths = 9
img_dir_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/data'
json_path = '/home/dl/wangleyuan/Rwork/Breast/dataset/BreastCancer/meta/2101.json'
# %%
all_path_list = glob(osp.join(img_dir_path, '*'))
path_list = np.random.choice(all_path_list, int(c_rate * len(all_path_list)))
print("num:{}/{}".format(len(path_list), len(all_path_list)))

# %%
imgs = []
labels = []
for x in tqdm(path_list):
    img = np.load(x)
    imgs.append(img)
    if img.mean((-1, -2)).mean() < 9:
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
param0 = [
    imgs[np.logical_not(labels)].mean((0, -1, -2)).tolist(),
    imgs[np.logical_not(labels)].std((0, -1, -2)).tolist()
]
print(param, '\n', param0, '\n', param1)
# %%
with open(json_path, 'r') as f:
    meta = json.load(f)
meta['mean_std'] = [param, param0, param1]
with open(json_path, 'w') as f:
    json.dump(meta, f)
# %%
