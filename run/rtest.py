import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.dataset import EyeDataset, get_eye_dataset
from src.rmodel import *
from src.util import SummaryWriter
from src.evaluation import r_evaluation, r_val_plot
from run.rtrain import set_r_model


def test(config, checkpoint):
    # data
    # print('Loading Data')
    test_dataset= checkpoint['cfg'][
        'dataset'] if config['dataset'] is None else config['dataset']
    test_data, _ = get_eye_dataset(datasets=[test_dataset],
                                   mode='val',
                                   less_data=config['less_data'])
    test_data_loader = DataLoader(test_data,
                                  config['batchsize'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=False,
                                  num_workers=config['num_workers'])
    if config['warmup']:
        warm_data, _ = get_eye_dataset(
            datasets=[test_dataset],
            mode='val',
            less_data=0.05,
        )
        warm_data_loader = DataLoader(warm_data,
                                      config['batchsize'],
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=config['num_workers'])

    # model and criterion
    # print('Load model')
    model = set_r_model(checkpoint['cfg'])
    cp_config = model.load_checkpoint(checkpoint)
    model.to_device(config['device'])

    # run
    model.eval()
    model.criterion = None

    if config['warmup']:
        model.init_val()
        # print('Warm up')
        with torch.no_grad():
            for warm_data in tqdm(warm_data_loader,
                                  ascii=False,
                                  dynamic_ncols=True):
                model.val_epoch(warm_data)

    model.init_val()
    # print('Run')
    with torch.no_grad():
        for test_data in tqdm(test_data_loader,
                              ascii=False,
                              dynamic_ncols=True):
            model.val_epoch(test_data)

    # evaluation
    val_save = model.val_save
    val_result, val_save = r_evaluation(val_save,
                                        len(test_data_loader.dataset), True)
    val_info = '\n'
    for k, v in val_result.items():
        if 'loss' in k.lower():
            val_info += " {}: {:.4e}\n".format(k, v)
        else:
            if isinstance(v, str):
                val_info += " {}: {}\n".format(k, v)
            else:
                val_info += " {}: {:.4f}\n".format(k, v)
    print(val_info)

    # write result
    if config['visible']:
        dst_dir = os.path.join("log", 'test', config['log_name'])
        log_writer = SummaryWriter(dst_dir)
        log_writer.add_text('cur_config', cp_config.__str__())
        log_writer.add_text('test_result', val_info)

        log_writer.add_hparams(
            {
                'train':
                str(cp_config['dataset']),
                'model': (cp_config['r_model_name']),
                'test':
                str(config['dataset'])
                if str(config['dataset']) else str(cp_config['dataset']),
                'lr':
                cp_config['lr'],
                'momentum':
                cp_config['momentum'],
                'weight_decay':
                cp_config['weight_decay'],
                'max_epoch':
                str(cp_config['r_max_epoch']),
                'bsize':
                cp_config['batchsize'],
                'device':
                str(cp_config['device'])
            },
            {
                'eer': val_result['eer'],
                'ch_index': val_result['ch_index'],
                # 'pred_loss': val_result['pred_loss'],
                'speed': val_result['speed'],
                'DI': val_result['DI'],
            })

        if r_val_plot is not None:
            r_val_plot(log_writer, 0, val_save, 'test')

        with open(os.path.join(dst_dir, 'log.txt'), 'w') as f:
            info = "test_config\n\n" + config.__str__()
            info += '\n' + '#' * 40 + '\n'
            info += "checkpoint_config\n\n" + cp_config.__str__()
            info += '\n' + '#' * 40 + '\n'
            info += "test_result\n" + val_info
            f.write(info)
        torch.save(val_save, os.path.join(dst_dir, 'save.pth'))
