import os
from scipy import stats
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.base_train import train
from src.util import LoadConfig
from src.dataset import FaceDataset
from src.model import AUnet, BUnet, CUnet
from src.framework import SegPredModel
from src.loss import *
from src.metric import SegmentationMetric, IQAMetric


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "CASIA-Complex-CX3, CUnet"
        self.log_name = "CX3"

        self.dataset = 'CASIA-Complex-CX3'
        self.cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = -1
        self.less_data = 0.125
        self.debug = False

        self.train_enhance = 0.5
        self.test_enhance = 0.3
        self.model_name = 'CUnet'
        self.img_down = 'bilinear'  # 'conv', 'nearest', 'bilinear', 'bicubic'
        self.att_down = 'avgpool'  # 'maxpool', 'avgpool', 'nearest', 'bilinear', 'bicubic'
        self.mask_loss = 'ce'  # ['sl1', 'l1', 'mse'], ['focal', 'dice', 'ce']
        self.gamma = (1, 0.05)  # (m,d)
        self.pred_loss = 'l1'

        self.batchsize = 8
        self.device = [2, 3]
        self.num_workers = 0
        self.seed = 2358

        self.frozen = 'None'

        self.max_epochs = 1500
        self.lr = 2e-2
        self.momentum = 0.9
        self.weight_decay = 1e-4

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

        self.log_name = "{}+{}+{}".format(
            self.model_name,
            self.mask_loss.upper(),
            self.log_name,
        )
        self.info = "{}, {}Loss, {}".format(
            self.model_name,
            self.mask_loss.upper(),
            self.info,
        )


def get_dataloaders(config):
    train_data = FaceDataset(dataset=config['dataset'],
                             mode='train',
                             enhance=config['train_enhance'],
                             resize=config['resize'] * 2,
                             less_data=config['less_data'],
                             mask_type=config['mask_type'])
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data = FaceDataset(
        dataset=config['dataset'],
        mode='val',
        enhance=config['test_enhance'],
        resize=config['resize'] * 2,
        #    less_data=config['less_data'],
        mask_type=config['mask_type'],
        load_sim=True)
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    return train_data_loader, val_data_loader


def evaluation(val_save, val_num):
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
    mask_loss = val_save['mask_loss'] / val_num
    sim = val_save['sim']
    name = np.concatenate(val_save['name'], axis=1)
    dfs = np.concatenate(val_save['dfs'], axis=0)
    pdfs = np.concatenate(val_save['pdfs'], axis=0)
    srocc = stats.spearmanr(pdfs.reshape(-1), dfs.reshape(-1))[0]
    lcc = stats.pearsonr(pdfs.reshape(-1), dfs.reshape(-1))[0]
    IRRs, EERs, auc = IQAMetric(sim, pdfs, name, 20)
    return {
        "Val_pred_loss": pred_loss,
        "Val_mask_loss": mask_loss,
        "AUC": auc,
        "mIoU": mIoU,
        "SROCC": srocc,
        "LCC": lcc
    }


def val_plot(log_writer, epoch, val_save):
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
        # mask[mask < 0.2] = 0
        # pmask[pmask < 0.2] = 0
        mask = mask.sum(1)
        pmask = pmask.sum(1)
    show_mask = torch.stack((pmask, mask, torch.zeros_like(pmask)), dim=1)
    mask += image[:, 0, ...]
    pmask += image[:, 0, ...]
    image = torch.clamp(torch.stack((pmask, mask, image[:, 0, ...]), dim=1), 0,
                        1)
    log_writer.add_images('Val/image', image, epoch)
    log_writer.add_images('Val/show_mask', show_mask, epoch)


if __name__ == "__main__":
    # set config
    config = Config()

    # data
    print('Loading Data')
    dataloaders = get_dataloaders(config)

    # criterion
    if config['frozen'] == 'seg':
        criterion = PDFSLoss(gamma=config['gamma'],
                             mask_loss=config['mask_loss'],
                             pred_loss=config['pred_loss'])
    elif config['mask_type'] == 'hard':
        criterion = HardMLoss(gamma=config['gamma'],
                              mask_loss=config['mask_loss'],
                              pred_loss=config['pred_loss'])
    elif config['mask_type'] == 'soft':
        criterion = SoftMLoss(gamma=config['gamma'],
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
    elif config['model_name'] == 'BUnet':
        model = BUnet(
            mask_type=config['mask_type'],
            img_down=[config['img_down'], config['resize']],
            att_down=config['att_down'],
        )
        model = SegPredModel(model, criterion)
    elif config['model_name'] == 'CUnet':
        model = CUnet(
            mask_type=config['mask_type'],
            img_down=[config['img_down'], config['resize']],
            att_down=config['att_down'],
        )
        model = SegPredModel(model, criterion)
    else:
        raise ValueError('Unsupported model: ' + config['model_name'])
    model.val_save['sim'] = dataloaders[1].dataset.sim

    # optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=config['log_interval'] * 2,
        verbose=True)

    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=config['lr'],
    #     momentum=0.9,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['max_epochs'] // 3,
    # )

    optimizers = (optimizer, scheduler)

    # train
    train(config, dataloaders, model, optimizers, evaluation, val_plot)
