import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import EyeDataset
from src.rmodel import *
from src.evaluation import r_evaluation, r_val_plot
from run.rtrain import set_model


def test(config, checkpoint):
    # data
    print('Loading Data')
    test_data = EyeDataset(dataset=checkpoint['cfg']['dataset'],
                           mode='test',
                           less_data=config['less_data'])
    test_data_loader = DataLoader(test_data,
                                  config['batchsize'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=False,
                                  num_workers=config['num_workers'])
    if config['warmup']:
        warm_data = EyeDataset(
            dataset=checkpoint['cfg']['dataset'],
            mode='test',
            less_data=0.05,
        )
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
    val_result = r_evaluation(
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
        log_writer = SummaryWriter(os.path.join("log", config['log_name']))
        log_writer.add_text('cur_config', cp_config.__str__())
        log_writer.add_text('test_result', val_info)
        if r_val_plot is not None:
            r_val_plot(log_writer, 0, val_save, 'test')
