import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.util import LoadConfig
from src.dataset import BaseDataset
from src.model import *
from src.evaluation import evaluation, val_plot
from train import set_model


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.log_name = ""

        self.cp_path = "checkpoints/0401_201634_siamese_vgg11bn_/5_8.0445e-01.pth"
        self.visible = True
        self.debug = False
        self.less_data = False
        self.warmup = False

        self.batchsize = 16
        self.device = [2]
        self.num_workers = 0
        self.seed = 2358

        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        self.log_name = "test_{}_{}".format(
            os.path.basename(os.path.dirname(self.cp_path)),
            self.log_name,
        )
        # if self.debug or self.less_data != False:
        #     self.warmup = False


if __name__ == "__main__":
    # set config
    config = Config()

    # load checkpoint
    checkpoint = torch.load(config['cp_path'],
                            map_location=torch.device('cpu'))

    # data
    print('Loading Data')
    test_data = BaseDataset(dataset=checkpoint['cfg']['dataset'],
                            mode='test',
                            less_data=config['less_data'],
                            resize=checkpoint['cfg']['resize'],
                            bg_ths=checkpoint['cfg']['bg_ths'],
                            bg_classifier=checkpoint['cfg']['bg_classifier'])
    test_data_loader = DataLoader(test_data,
                                  config['batchsize'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=False,
                                  num_workers=config['num_workers'])
    if config['warmup']:
        warm_data = BaseDataset(
            dataset=checkpoint['cfg']['dataset'],
            mode='test',
            less_data=0.05,
            resize=checkpoint['cfg']['resize'],
            bg_ths=checkpoint['cfg']['bg_ths'],
            bg_classifier=checkpoint['cfg']['bg_classifier'])
        warm_data_loader = DataLoader(warm_data,
                                      config['batchsize'],
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=config['num_workers'])

    # model and criterion
    print('Load model')
    model = set_model(checkpoint['cfg'])
    cp_config = model.load_checkpoint(checkpoint)
    model.to_device(config['device'])

    # run
    model.eval()

    if config['warmup']:
        model.init_val()
        print('Warm up')
        with torch.no_grad():
            for warm_data in tqdm(warm_data_loader,
                                  ascii=False,
                                  dynamic_ncols=True):
                model.val_epoch(warm_data)

    model.init_val()
    print('Run')
    with torch.no_grad():
        for test_data in tqdm(test_data_loader,
                              ascii=False,
                              dynamic_ncols=True):
            model.val_epoch(test_data)

    # evaluation
    val_save = model.val_save
    val_result = evaluation(
        val_save,
        len(test_data_loader.dataset),
    )
    val_info = '\n'
    for k, v in val_result.items():
        if 'loss' in k.lower():
            val_info += " {}: {:.4e}\n".format(k, v)
        else:
            val_info += " {}: {:.4f}\n".format(k, v)
    print(val_info)

    # write result
    if config['visible']:
        # cp_dir_path = os.path.normcase(
        #     os.path.join('checkpoints', config['log_name']))
        # os.mkdir(cp_dir_path)
        # with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
        #     if config['cp_path']:
        #         result = str(
        #             config) + '#' * 30 + '\ncheckpoint_config:\n' + str(
        #                 cp_config) + '#' * 30 + '\n'
        #     else:
        #         result = str(config) + '#' * 30 + '\n'
        #     result += val_info
        #     f.write(result)
        log_writer = SummaryWriter(os.path.join("log", config['log_name']))
        log_writer.add_text('cur_config', cp_config.__str__())
        log_writer.add_text('test_result', val_info)
        if val_plot is not None:
            val_plot(log_writer, 0, val_save, 'test')