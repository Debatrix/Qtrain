import os
import cv2
from scipy import stats
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.util import LoadConfig
from src.dataset import FaceDataset
from src.model import AUnet
from src.framework import SegPredModel
from src.loss import *
from src.metric import IQAMetric, SegmentationMetric


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = ''
        self.log_name = "CX3"

        self.dataset = 'CASIA-Iris-Distance'
        self.cp_path = "checkpoints/0108_224158_AUnet+CE+CX3-alldata/100_8.2286e-04.pth"
        self.visible = True
        self.less_data = 1
        self.bins = 50

        self.test_enhance = 0
        self.model_name = 'AUnet'
        self.img_down = 'bilinear'  # 'conv', 'nearest', 'bilinear', 'bicubic'
        self.att_down = 'avgpool'  # 'maxpool', 'avgpool', 'nearest', 'bilinear', 'bicubic'
        self.mask_loss = 'ce'  # ['sl1', 'l1', 'mse'], ['focal', 'dice', 'ce']
        self.pred_loss = 'mse'

        self.batchsize = 8
        self.device = [2, 3]
        self.num_workers = 0
        self.seed = 45684

        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        self.resize = 4 if self.dataset == 'CASIA-Complex-CX3' else 2
        if self.mask_loss.lower() in ['sl1', 'l1', 'mse']:
            self.mask_type = 'soft'
        elif self.mask_loss.lower() in ['focal', 'dice', 'ce']:
            self.mask_type = 'hard'
        else:
            raise ValueError('Unsupported mask loss ' + self.mask_loss)


def evaluation(val_save, val_num, bins=50):
    metric = SegmentationMetric(2)
    mask = val_save['mask']
    pmask = val_save['pmask']
    if config['mask_type'] == 'soft':
        mask[mask < 0.2] = 0
        mask[mask >= 0.2] = 1
        mask = mask[:, 0, :, :] + 2 * mask[:, 1, :, :]
        bg = torch.ones(
            (pmask.shape[0], pmask.shape[2], pmask.shape[3])) - pmask.sum(1)
        pmask = torch.softmax(torch.cat((bg.unsqueeze(1), pmask), 1), 1)
    metric.update(pmask, mask)
    pixAcc, mIoU = metric.get()

    pred_loss = val_save['pred_loss'] / val_num
    sim = val_save['sim']
    name = np.concatenate(val_save['name'], axis=1)
    dfs = np.concatenate(val_save['dfs'], axis=0)
    pdfs = np.concatenate(val_save['pdfs'], axis=0)
    srocc = stats.spearmanr(pdfs.reshape(-1), dfs.reshape(-1))[0]
    lcc = stats.pearsonr(pdfs.reshape(-1), dfs.reshape(-1))[0]
    speed = val_num / val_save['all_time']
    IRRs, EERs, auc = IQAMetric(sim, pdfs, name, bins)
    val_save['irr-eer'] = (IRRs, EERs)
    return val_save, {
        "Val_pred_loss": pred_loss,
        "mIoU": mIoU,
        "SROCC": srocc,
        "LCC": lcc,
        "AUC": auc,
        'speed': speed
    }


if __name__ == "__main__":
    # set config
    config = Config()

    # data
    print('Loading Data')
    val_data = FaceDataset(
        dataset=config['dataset'],
        mode='test',
        enhance=config['test_enhance'],
        resize=config['resize'] * 2,
        less_data=config['less_data'],
        mask_type=config['mask_type'],
        load_sim=True,
    )
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=False,
                                 num_workers=config['num_workers'])

    # criterion
    if config['mask_type'] == 'hard':
        criterion = HardMLoss(gamma=(1, 1),
                              mask_loss=config['mask_loss'],
                              pred_loss=config['pred_loss'])
    elif config['mask_type'] == 'soft':
        criterion = SoftMLoss(gamma=(1, 1),
                              mask_loss=config['mask_loss'],
                              pred_loss=config['pred_loss'])
    else:
        raise ValueError('Unsupported mask type: ' + config['mask_type'])

    # model
    if config['model_name'] == 'AUnet':
        model = AUnet(
            mask_type=config['mask_type'],
            img_down=[config['img_down'], config['resize']],
            att_down=config['att_down'],
        )
        model = SegPredModel(model, criterion)
    else:
        raise ValueError('Unsupported model: ' + config['model_name'])
    model.val_save['sim'] = val_data_loader.dataset.sim

    cp_config = model.load_checkpoint(config['cp_path'])
    model.to_device(config['device'])

    # run
    model.eval()
    model.init_val()
    with torch.no_grad():
        for val_data in tqdm(val_data_loader, ):
            model.val_epoch(val_data)

    # evaluation
    val_save = model.val_save
    val_save, val_result = evaluation(val_save, len(val_data_loader),
                                      config['bins'])
    val_info = '\n'
    for k, v in val_result.items():
        if 'loss' in k.lower():
            val_info += " {}: {:.4e}\n".format(k, v)
        else:
            val_info += " {}: {:.4f}\n".format(k, v)
    print(val_info)

    # write result
    if config['visible']:
        cp_dir_path = os.path.normcase(
            os.path.join('checkpoints', 'test_' + config['log_name']))
        os.mkdir(cp_dir_path)
        with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
            if config['cp_path']:
                result = str(
                    config) + '#' * 30 + '\ncheckpoint_config:\n' + str(
                        cp_config) + '#' * 30 + '\n'
            else:
                result = str(config) + '#' * 30 + '\n'
            result += val_info
            f.write(result)

        plt.plot(*val_save['irr-eer'])
        plt.xlabel('irr')
        plt.ylabel('eer')
        plt.savefig(os.path.join(cp_dir_path, 'irr-eer.png'))

        image = nn.functional.interpolate(
            val_save['image'],
            (val_save['mask'][0].shape[-2], val_save['mask'][0].shape[-1]),
            mode='bilinear',
            align_corners=True)
        mask = val_save['mask']
        pmask = val_save['pmask']
        if config['mask_type'] == 'hard':
            mask[mask != 0] = 1
            mask = mask.to(torch.float)
            pmask = torch.nn.functional.softmax(pmask, 1)
            pmask = pmask[:, 1:, :, :].sum(1)
        else:
            mask = mask.sum(1)
            pmask = pmask.sum(1)
        show_mask = torch.stack((torch.zeros_like(mask), mask, pmask),
                                dim=1).numpy().transpose((0, 2, 3, 1))
        mask += image[:, 0, ...] * 0.75
        pmask += image[:, 0, ...] * 0.75
        image = torch.clamp(
            torch.stack((image[:, 0, ...] * 0.75, mask, pmask), dim=1), 0, 1)
        image = image.numpy().transpose((0, 2, 3, 1))
        for x in range(image.shape[0]):
            cv2.imwrite(os.path.join(cp_dir_path, '{}.png'.format(x)),
                        np.round(image[x] * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(cp_dir_path, '{}_mask.png'.format(x)),
                        np.round(show_mask[x] * 255).astype(np.uint8))
