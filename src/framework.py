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


class PredictModel(BaseModel):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(PredictModel, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['label'] = []
        self.val_save['prediction'] = []
        self.val_save['feature'] = []
        self.val_save['image'] = []
        self.val_save['instance'] = []
        self.val_save['tag'] = []
        self.val_save['all_time'] = 0

    def _feed_data(self, input):
        img, label = input['img'], input['label']
        if not self.is_cpu:
            img = img.cuda()
            label = label.cuda()
        return img, label

    def train_epoch(self, input):
        img, label = self._feed_data(input)
        output = self.model(img)
        output['label'] = label
        loss_dict = self.criterion(output)
        return loss_dict

    def val_epoch(self, input):
        img, label = self._feed_data(input)

        start = time.time()
        output = self.model(img)
        self.val_save['all_time'] += time.time() - start

        output['label'] = label
        loss_dict = self.criterion(output)

        self.val_save['pred_loss'] += loss_dict['pred_loss'].cpu().item()
        self.val_save['prediction'].append(output['prediction'].cpu().numpy())
        self.val_save['label'].append(label.cpu().numpy())
        self.val_save['feature'].append(output['feature'].cpu().numpy())
        self.val_save['instance'] += input['instance']
        self.val_save['tag'] += input['tag']
        return self.val_save


class SiameseModel(BaseModel):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(SiameseModel, self).__init__(model, criterion)
        self.classifier = svm.SVC(kernel='rbf')
        self.train_save = {}

    def init_train(self):
        self.train_save['label'] = []
        self.train_save['feature'] = []

    def init_val(self):
        self.val_save['proba'] = []
        self.val_save['label'] = []
        self.val_save['prediction'] = []
        self.val_save['feature'] = []
        self.val_save['image'] = []
        self.val_save['instance'] = []
        self.val_save['tag'] = []
        self.val_save['all_time'] = 0

    def _feed_paired_data(self, input):
        img1, label1 = input['img1'], input['label1']
        img2, label2 = input['img2'], input['label2']
        flag = input['flag']
        if not self.is_cpu:
            img1 = img1.cuda()
            label1 = label1.cuda()
            img2 = img2.cuda()
            label2 = label2.cuda()
            flag = flag.cuda()
        return img1, label1, img2, label2, flag

    def train_epoch(self, input):
        output = {}
        img1, label1, img2, label2, output['flag'] = self._feed_paired_data(
            input)

        _output1 = self.model(img1)
        _output2 = self.model(img2)

        output['label1'] = label1
        output['label2'] = label2
        output['feature1'] = _output1['feature']
        output['feature2'] = _output2['feature']
        output['prediction1'] = _output1['prediction']
        output['prediction2'] = _output2['prediction']
        loss_dict = self.criterion(output)

        self.train_save['label'].append(label1.cpu().numpy())
        self.train_save['feature'].append(
            _output1['feature'].detach().cpu().numpy())

        return loss_dict

    def train_finish(self):
        label = np.concatenate(self.train_save['label'], axis=0)
        feature = np.concatenate(self.train_save['feature'], axis=0)
        self.classifier.fit(feature, label)

    def _feed_data(self, input):
        img, label = input['img'], input['label']
        if not self.is_cpu:
            img = img.cuda()
            label = label.cuda()
        return img, label

    def val_epoch(self, input):
        img, label = self._feed_data(input)

        start = time.time()
        output = self.model(img)
        self.val_save['all_time'] += time.time() - start

        output['label'] = label
        feature = output['feature'].cpu().numpy()
        proba = self.classifier.decision_function(feature)
        prediction = self.classifier.predict(feature)

        # self.val_save['pred_loss'] += loss_dict['pred_loss'].cpu().item()
        self.val_save['prediction'].append(prediction)
        self.val_save['proba'].append(proba)
        self.val_save['label'].append(label.cpu().numpy())
        self.val_save['feature'].append(feature)
        self.val_save['instance'] += input['instance']
        self.val_save['tag'] += input['tag']
        return self.val_save

    def load_checkpoint(self, cp_path=None):
        cp_config = None
        if cp_path:
            if isinstance(cp_path, str):
                cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
            elif isinstance(cp_path, dict):
                cp_data = cp_path
            assert 'model' in cp_data
            assert 'cfg' in cp_data
            assert 'classifier' in cp_data
            try:
                self.model.load_state_dict(cp_data['model'])
            except Exception as e:
                self.model.load_state_dict(cp_data['model'], strict=False)
                print(e)
            self.classifier = cp_data['classifier']
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
            classifier=self.classifier,
        )
        torch.save(cp_data, save_path)