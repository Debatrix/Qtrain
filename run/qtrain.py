from tqdm import tqdm
import torch

from src.base_train import train as base_train
from src.evaluation import r_evaluation, q_val_plot, q_evaluation

from run.set_model import set_eye_dataloaders, set_face_dataloaders, set_q_model, set_r_model


def prepare(config):
    dfs = []
    data_loader, _ = set_eye_dataloaders(config, 'qtrain')
    train_data_loader, val_data_loader = data_loader

    checkpoint = torch.load(config['r_cp_path'],
                            map_location=torch.device('cpu'))
    model = set_r_model(checkpoint['cfg'])
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
    val_result, val_save = r_evaluation(
        val_save,
        len(train_data_loader.dataset),
    )
    dfs.append(val_save['dfs'])

    model.init_val()
    with torch.no_grad():
        for test_data in tqdm(val_data_loader, ascii=False,
                              dynamic_ncols=True):
            model.val_epoch(test_data)
    val_save = model.val_save
    val_result, val_save = r_evaluation(
        val_save,
        len(val_data_loader.dataset),
    )
    dfs.append(val_save['dfs'])

    return dfs


def train(config):
    dfs = prepare(config)

    # data
    print('Loading Data')
    dataloaders = set_face_dataloaders(config, 'qtrain', dfs)

    # model and
    model = set_q_model(config)

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
    base_train(config, dataloaders, model, optimizers, q_evaluation,
               q_val_plot)