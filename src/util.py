import os
import os.path as osp
import torch.nn.functional as F
from datetime import datetime
from argparse import ArgumentParser
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class LoadConfig(object):
    def __init__(self):
        self.info = ""
        self.log_name = ""

        self.dataset = ''
        self.cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = 20
        self.less_data = False
        self.debug = False

        self.batchsize = 8
        self.device = [0, 1, 2, 3]
        self.num_workers = 2
        self.seed = np.random.randint(9999)

        self.max_epochs = 500
        self.lr = 2e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4

    def apply(self):
        parser = ArgumentParser()
        for name, value in self.__dict__.items():
            parser.add_argument('--' + name, type=type(value), default=value)
        args = parser.parse_args()

        for name, value in vars(args).items():
            if self.__dict__[name] != value:
                self.__dict__[name] = value
        self.change_setting()

        self.git_hash = os.popen('git rev-parse --short HEAD').read()

    def change_setting(self):
        if self.log_name:
            self.log_name = get_datestamp() + '_' + self.log_name
        else:
            self.log_name = get_datestamp()

        if self.debug:
            self.save_interval = 1
            self.log_interval = 1
            self.max_epochs = 2
            self.log_name += '_debug'
        else:
            import warnings
            warnings.filterwarnings("ignore")

        if len(self.device) > 1:
            self.batchsize = self.batchsize * len(self.device)
            self.num_workers *= len(self.device)

        device = ','
        device = device.join([str(x) for x in self.device])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    def __str__(self):
        config = ""
        for name, value in self.__dict__.items():
            config += ('%s=%s\n' % (name, value))
        return config

    def __getitem__(self, key):
        # if key not in self.__dict__:
        #     raise AttributeError(
        #         "'LoadConfig' object has no attribute '{}'".format(key))
        return self.__dict__.get(key)

    def __contains__(self, item):
        return item in self.__dict__

    def get_dict(self):
        return self.__dict__


####################
# miscellaneous
####################

tqdm_bar = '{desc} {n_fmt}/{total_fmt}-{percentage:3.0f}%|{rate_fmt}'


def get_datestamp():
    return datetime.now().strftime('%m%d_%H%M%S')


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


####################
# Save and load
####################
def save_model(model, save_path, info=None):
    if isinstance(model, nn.DataParallel) or isinstance(
            model, DistributedDataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    cp_data = dict(
        cfg=info,
        model=state_dict,
    )
    torch.save(cp_data, save_path)


def load_model(load_path, model, strict=True):
    if isinstance(model, nn.DataParallel) or isinstance(
            model, DistributedDataParallel):
        model = model.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    model.load_state_dict(load_net_clean, strict=strict)


class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError(
                'hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
