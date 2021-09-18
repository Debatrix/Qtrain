# %%
import os.path as osp

import torch
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# %%
thousand = torch.load(
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183515_0624_232726_r_maxout_o_thousand_0_pth_thousand/save.pth'
)
distance = torch.load(
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183056_0624_232827_r_maxout_o_distance_0_pth_distance/save.pth'
)

# %%
t_norm = np.linalg.norm(thousand['feature'], axis=1)
d_norm = np.linalg.norm(distance['feature'], axis=1)

# %%
tn, tbins, tpatches = plt.hist(t_norm,
                               50,
                               density=True,
                               facecolor='g',
                               alpha=0.75,
                               label='CASIA-Iris-Thousand')
dn, dbins, dpatches = plt.hist(d_norm,
                               50,
                               density=True,
                               facecolor='r',
                               alpha=0.75,
                               label='CASIA-Iris-Distance')
plt.xlim(10, 60)
plt.xlabel('L2-norm')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
# %%
cp_path = {
    'tdor':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_181249_0624_232749_r_maxout_o_thousand_0_01_pth_distance',
    'ttor':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_181316_0624_232749_r_maxout_o_thousand_0_01_pth_thousand',
    'tdnr':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_181804_0624_233040_r_maxout_thousand_0_01_pth_distance',
    'ttnr':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_181815_0624_233040_r_maxout_thousand_0_01_pth_thousand',
    'tdno':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_182224_0624_233023_r_maxout_thousand_0_pth_distance',
    'ttno':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_182235_0624_233023_r_maxout_thousand_0_pth_thousand',
    'ddno':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_182646_0630_171312_r_maxout_distance_0_pth_distance',
    'dtno':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_182657_0630_171312_r_maxout_distance_0_pth_thousand',
    'ddoo':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183056_0624_232827_r_maxout_o_distance_0_pth_distance',
    'dtoo':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183107_0624_232827_r_maxout_o_distance_0_pth_thousand',
    'tdoo':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183504_0624_232726_r_maxout_o_thousand_0_pth_distance',
    'ttoo':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183515_0624_232726_r_maxout_o_thousand_0_pth_thousand',
    'ddor':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183924_0624_232810_r_maxout_o_distance_0_01_pth_distance',
    'dtor':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183934_0624_232810_r_maxout_o_distance_0_01_pth_thousand',
    'ddnr':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183934_0624_232810_r_maxout_o_distance_0_01_pth_thousand',
    'dtnr':
    '/home/leyuan.wang/Rwork/Qtrain/log/test/0630_183934_0624_232810_r_maxout_o_distance_0_01_pth_thousand'
}

# %%
rocs = {}
for k, v in cp_path.items():
    print(v)
    rocs[k] = torch.load(v + '/save.pth')['roc']

# %%
plt.plot(rocs['ttoo'][0],
         rocs['ttoo'][1],
         color='r',
         label='without norm, without reg')
plt.plot(rocs['ttor'][0],
         rocs['ttor'][1],
         color='m',
         label='without norm, with reg')
plt.plot(rocs['ttno'][0],
         rocs['ttno'][1],
         color='b',
         label='with norm, without reg')
plt.plot(rocs['ttnr'][0],
         rocs['ttnr'][1],
         color='g',
         label='with norm, with reg')
plt.grid()
plt.xscale("log")
plt.xlabel('False Positive Rate (in log scale)')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# %%
plt.plot(rocs['ttoo'][0],
         rocs['ttoo'][1],
         color='g',
         label='Thousand(eer:6.64%)')
plt.plot(rocs['ddoo'][0],
         rocs['ddoo'][1],
         color='r',
         label='Distance(eer:12.00%)')

plt.grid()
plt.xscale("log")
plt.xlabel('False Positive Rate (in log scale)')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# %%
plt.plot(rocs['ddoo'][0],
         rocs['ddoo'][1],
         color='r',
         label='without norm, without reg')
plt.plot(rocs['ddor'][0],
         rocs['ddor'][1],
         color='m',
         label='without norm, with reg')
plt.plot(rocs['ddno'][0],
         rocs['ddno'][1],
         color='b',
         label='with norm, without reg')
plt.plot(rocs['ddnr'][0],
         rocs['ddnr'][1],
         color='g',
         label='with norm, with reg')
plt.grid()
plt.xscale("log")
plt.xlabel('False Positive Rate (in log scale)')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# %%
