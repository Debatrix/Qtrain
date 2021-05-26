import os
from tqdm import tqdm

import torch

from src.base_train import base_train_head, train as base_train, train_body
from src.evaluation import r_evaluation, r_val_plot, q_evaluation
from run.set_model import set_eye_dataloaders, set_face_dataloaders, set_q_model, set_r_model


def prepare(config):
    dfs = []
    train_data_loader, _ = set_face_dataloaders(config, 'rtrain')

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
    val_result, val_save = q_evaluation(
        val_save,
        len(train_data_loader.dataset),
    )
    dfs = val_save['pdfs']

    return dfs


def _train(config, model, dataloaders, log_writer):
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
    model = train_body(
        config,
        model,
        dataloaders,
        optimizers,
        r_evaluation,
        log_writer,
        r_val_plot,
    )
    return model


def train(config):
    if config['q_cp_path']:
        pdfs = prepare(config)
    else:
        pdfs = None
    # data
    print('Loading Data')
    dataloaders, num_classes = set_eye_dataloaders(config, 'rtrain', pdfs)
    config['num_classes'] = num_classes

    # model
    model = set_r_model(config)
    config, model, log_writer = base_train_head(config, model)
    for round in range(1, config['r_max_epoch'][0] + 1):
        config['tag'] = 'R{}'.format(round)
        config['cur_epoch'] = (round - 1) * config['r_max_epoch'][1]
        config['max_epochs'] = round * config['r_max_epoch'][1]

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

        optimizer = torch.optim.SGD(
            params,
            lr=config['lr'],
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.5,
            patience=config['log_interval'] * 2,
            verbose=True)

        # train
        model = train_body(
            config,
            model,
            dataloaders,
            (optimizer, scheduler),
            r_evaluation,
            log_writer,
            r_val_plot,
        )
        save_path = os.path.join(config['cp_dir_path'],
                                 'Round{}.pth'.format(round))
        model.save_checkpoint(save_path, info=config)
