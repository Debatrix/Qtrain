from run.set_model import set_q_model, set_r_model
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.base_train import train as base_train
from src.dataset import EyeDataset, FaceDataset
from src.evaluation import r_evaluation, r_val_plot, q_evaluation


def prepare(config):
    dfs = []
    train_data = FaceDataset(
        dataset=config['dataset'],
        mode='rtrain',
        less_data=config['less_data'],
    )
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])

    checkpoint = torch.load(config['q_cp_path'],
                            map_location=torch.device('cpu'))
    model = set_q_model(checkpoint['cfg'])
    _ = model.load_checkpoint(checkpoint)
    model.to_device(config['device'])

    # run
    print('Preparing dfs')
    model.eval()
    model.init_val()
    with torch.no_grad():
        for test_data in tqdm(train_data_loader,
                              ascii=False,
                              dynamic_ncols=True):
            model.val_epoch(test_data)
    val_save = model.val_save
    val_result = q_evaluation(
        val_save,
        len(train_data_loader.dataset),
    )
    dfs = val_save['pdfs']

    return dfs


def set_dataloaders(config, dfs):
    train_data = EyeDataset(dataset=config['dataset'],
                            mode='rtrain',
                            less_data=config['less_data'],
                            dfs=dfs,
                            weight=config['weight'])
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data = EyeDataset(dataset=config['dataset'],
                          mode='val',
                          less_data=config['less_data'])
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    num_classes = train_data.num_classes
    return (train_data_loader, val_data_loader), num_classes


def train(config):
    if config['q_cp_path']:
        dfs = prepare(config)
    else:
        dfs = [None, None]
    # data
    print('Loading Data')
    dataloaders, num_classes = set_dataloaders(config, dfs)
    config.num_classes = num_classes

    # model and
    model = set_r_model(config)

    # optimizer and scheduler
    params = []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'bias' in name:
            params += [{
                'params': value,
                'lr': 2 * config['lr'],
                'weight_decay': 0
            }]
        else:
            params += [{'params': value, 'lr': config['lr']}]

    optimizer = torch.optim.Adam(
        params,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=config['log_interval'] * 2,
        verbose=True)

    # optimizer = torch.optim.SGD(
    #     params,
    #     lr=config['lr'],
    #     momentum=0.9,
    # )
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     'min',
    #     factor=0.5,
    #     patience=config['log_interval'] * 2,
    #     verbose=True)

    optimizers = (optimizer, scheduler)

    # train
    base_train(config, dataloaders, model, optimizers, r_evaluation,
               r_val_plot)