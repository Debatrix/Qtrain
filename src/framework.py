import time
import numpy as np
from sklearn import svm

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


class BaseModel(nn.Module):
    def __init__(self, model, criterion):
        super(BaseModel, self).__init__()
        self.model = model
        self.val_save = {}
        self.criterion = criterion

    def to_device(self, device=['cpu']):
        if isinstance(device, str):
            device = [device]

        if torch.cuda.is_available() and device[0] is not 'cpu':
            # torch.cuda.set_device('cuda:{}'.format(device[0]))
            _device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
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
            if isinstance(cp_path, str):
                cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
            elif isinstance(cp_path, dict):
                cp_data = cp_path
            assert 'model' in cp_data
            assert 'cfg' in cp_data
            try:
                self.model.load_state_dict(cp_data['model'])
            except Exception as e:
                self.model.load_state_dict(cp_data['model'], strict=False)
                print(e)
            cp_config = '' if 'cfg' not in cp_data else cp_data['cfg']
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

    def init_train(self):
        pass

    def train_epoch(self, input):
        raise NotImplementedError

    def val_epoch(self, input):
        raise NotImplementedError

    def train_finish(self):
        pass


class RecognitionModel(BaseModel):
    def __init__(self, model, criterion=nn.CrossEntropyLoss()):
        super(RecognitionModel, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['label'] = []
        self.val_save['name'] = []
        self.val_save['prediction'] = []
        self.val_save['feature'] = []
        self.val_save['all_time'] = 0

    def _feed_data(self, input):
        img, label, weight = input['img'], input['label'], input['weight']
        if not self.is_cpu:
            img = img.cuda()
            label = label.cuda()
            weight = weight.cuda()
        return img, label, weight

    def train_epoch(self, input):
        img, label, weight = self._feed_data(input)
        output = self.model(img)
        output['label'] = label
        output['weight'] = weight
        loss_dict = self.criterion(output)
        return loss_dict

    def val_epoch(self, input):

        start = time.time()
        img, label, _ = self._feed_data(input)
        output = self.model(img)
        spend = time.time() - start
        self.val_save['all_time'] += spend

        output['label'] = label
        # if self.criterion:
        #     loss_dict = self.criterion(output)
        #     self.val_save['pred_loss'] += loss_dict['pred_loss'].cpu().item()

        self.val_save['prediction'].append(output['prediction'].cpu().numpy())
        self.val_save['label'].append(label.cpu().numpy())
        self.val_save['feature'].append(output['feature'].cpu().numpy())
        self.val_save['name'].append(input['name'])
        return self.val_save


class TripRecognitionModel(RecognitionModel):
    def __init__(self, model, criterion=nn.TripletMarginLoss()):
        super(TripRecognitionModel, self).__init__(model, criterion)

    def _feed_train_data(self, input):
        img1, img2, img3 = input['img1'], input['img2'], input['img3']
        img = torch.cat((img1, img2, img3), 0)
        if not self.is_cpu:
            img = img.cuda()
        return img

    def train_epoch(self, input):
        img = self._feed_train_data(input)
        output = self.model(img)
        loss_dict = self.criterion(output)
        return loss_dict


class IQAnModel(BaseModel):
    def __init__(self, model, criterion=nn.CrossEntropyLoss()):
        super(IQAnModel, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['dfs'] = []
        self.val_save['name'] = []
        self.val_save['mask'] = []
        self.val_save['heatmap'] = []
        self.val_save['image'] = []
        self.val_save['prediction'] = []
        self.val_save['all_time'] = 0

    def _feed_data(self, input):
        img, mask, dfs = input['img'], input['mask'], input['dfs']
        if not self.is_cpu:
            img = img.cuda()
            mask = mask.cuda()
            dfs = dfs.cuda()
        return img, mask, dfs

    def train_epoch(self, input):
        img, mask, dfs = self._feed_data(input)
        output = self.model(img)
        output['mask'] = mask
        output['dfs'] = dfs
        loss_dict = self.criterion(output)
        return loss_dict

    def val_epoch(self, input):
        img, mask, dfs = self._feed_data(input)

        start = time.time()
        output = self.model(img)
        output['mask'] = mask
        output['dfs'] = dfs
        self.val_save['all_time'] += time.time() - start

        loss_dict = self.criterion(output)

        self.val_save['pred_loss'] += loss_dict['pred_loss'].cpu().item()
        self.val_save['prediction'].append(output['prediction'].cpu().numpy())
        self.val_save['heatmap'].append(output['heatmap'].cpu().numpy())
        self.val_save['dfs'].append(dfs.cpu().numpy())
        self.val_save['mask'].append(mask.cpu().numpy())
        self.val_save['image'].append(img.cpu().numpy())
        self.val_save['name'].append(np.array(input['ename']).T)
        return self.val_save