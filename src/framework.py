import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def init_params(self, scale=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class base_model(nn.Module):
    def __init__(self, model, criterion):
        super(base_model, self).__init__()
        self.model = model
        self.val_save = {}
        self.criterion = criterion

    def to_device(self, device=['cpu']):
        if isinstance(device, str):
            device = [device]

        if torch.cuda.is_available() and device[0] is not 'cpu':
            # torch.cuda.set_device('cuda:{}'.format(device[0]))
            _device = torch.device('cuda:0')
            self.is_cpu = False
        else:
            if not torch.cuda.is_available():
                print("hey man, buy a GPU!")
            _device = torch.device('cpu')
            self.is_cpu = True

        self.model = self.model.to(_device)
        self.criterion = self.criterion.to(
            _device) if self.criterion is not None else None
        if len(device) > 1:
            self.model = nn.DataParallel(self.model)

        return self

    def load_checkpoint(self, cp_path=None):
        cp_config = None
        if cp_path:
            cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
            try:
                self.model.load_state_dict(cp_data['model'])
            except Exception as e:
                self.model.load_state_dict(cp_data['model'], strict=False)
                print(e)
            cp_config = '' if 'config' not in cp_data else cp_data['config']
            # print('Load checkpoint {}'.format(
            #     {x.split('=')[0]: x
            #      for x in cp_config.split('\n')}['log_name']))
        return cp_config

    def save_checkpoint(self, save_path, info=None):
        if isinstance(self.model, nn.DataParallel) or isinstance(
                self.model, DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        cp_data = dict(
            cfg=info,
            model=state_dict,
        )
        torch.save(cp_data, save_path)

    def load_state_dict(self, checkpoint):
        return self.model.load_state_dict(checkpoint)

    def _feed_data(self, input):
        raise NotImplementedError

    def init_val(self):
        raise NotImplementedError

    def train_epoch(self, input):
        raise NotImplementedError

    def val_epoch(self, input):
        raise NotImplementedError


class SegPredModel(base_model):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(SegPredModel, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['mask_loss'] = 0
        self.val_save['pdfs'] = []
        self.val_save['dfs'] = []
        self.val_save['image'] = []
        self.val_save['mask'] = []
        self.val_save['pmask'] = []
        self.val_save['name'] = []
        self.val_save['all_time'] = 0

    def _feed_data(self, input):
        img, mask, dfs = input['img'], input['mask'], input['dfs']
        if not self.is_cpu:
            img = img.cuda()
            dfs = dfs.cuda()
            mask = mask.cuda()
        return img, mask, dfs

    def train_epoch(self, input):
        img, mask, dfs = self._feed_data(input)
        loss_input = {'dfs': dfs, 'mask': mask}
        output = self.model(img)
        loss_input['pdfs'], loss_input['pmask'] = output[:2]
        loss_dict = self.criterion(loss_input)
        return loss_dict

    def val_epoch(self, input):
        img, mask, dfs = self._feed_data(input)
        start = time.time()
        output = self.model(img)
        self.val_save['all_time'] += time.time() - start
        pdfs, pmask = output[:2]
        loss_input = {
            'dfs': dfs,
            'mask': mask,
            'pdfs': pdfs,
            'pmask': pmask,
        }
        loss_dict = self.criterion(loss_input)

        self.val_save['pred_loss'] += loss_dict['pred_loss']
        self.val_save['mask_loss'] += loss_dict['mask_loss']
        self.val_save['pdfs'].append(pdfs.cpu().numpy())
        self.val_save['dfs'].append(dfs.cpu().numpy())
        self.val_save['name'].append(np.array(input['name']))
        if len(self.val_save['image']) <= img.shape[0]:
            self.val_save['image'] = img.cpu()
            self.val_save['mask'] = mask.cpu()
            self.val_save['pmask'] = pmask.cpu()

        return self.val_save