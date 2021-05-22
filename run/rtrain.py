import torch
from torch.utils.data import DataLoader

from src.base_train import train as base_train
from src.dataset import EyeDataset
from src.rmodel import *
from src.framework import RecognitionModel
from src.loss import PredictLoss
from src.evaluation import r_evaluation, r_val_plot


def set_dataloaders(config):
    train_data = EyeDataset(
        dataset=config['dataset'],
        mode='rtrain',
        less_data=config['less_data'],
    )
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data = EyeDataset(
        dataset=config['dataset'],
        mode='val',
        less_data=config['less_data'],
    )
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    num_classes = train_data.num_classes
    return (train_data_loader, val_data_loader), num_classes


def set_model(config):

    # model
    if config['model_name'].lower() == 'maxout':
        model = Maxout(num_classes=config['num_classes'])
    elif config['model_name'].lower() == 'resnet18':
        model = Resnet18(num_classes=config['num_classes'],
                         pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'vgg11bn':
        model = VGG11BN(num_classes=config['num_classes'],
                        pretrained=config['pretrained'])
    else:
        raise ValueError('Unsupported model: ' + config['model_name'])

    # criterion
    criterion = PredictLoss(pred_loss=config['pred_loss'])
    model = RecognitionModel(model, criterion)

    return model


def train(config):
    # data
    print('Loading Data')
    dataloaders, num_classes = set_dataloaders(config)
    config.num_classes = num_classes

    # model and
    model = set_model(config)

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

    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, params),
    #     lr=config['lr'],
    #     weight_decay=config['weight_decay'],
    # )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     'min',
    #     factor=0.5,
    #     patience=config['log_interval'] * 2,
    #     verbose=True)

    optimizer = torch.optim.SGD(
        params,
        lr=config['lr'],
        momentum=0.9,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=config['log_interval'] * 2,
        verbose=True)

    optimizers = (optimizer, scheduler)

    # train
    base_train(config, dataloaders, model, optimizers, r_evaluation,
               r_val_plot)