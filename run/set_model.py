import torch
from torch.utils.data import DataLoader

from src import qmodel, rmodel
from src.loss import IQALoss, PredictLoss
from src.framework import IQAnModel, RecognitionModel
from src.dataset import EyeDataset, FaceDataset


def set_r_model(config):
    if 'r_model_name' in config:
        model_name = config['r_model_name']
    else:
        model_name = config['model_name'].split('_')[-1]

    # model
    if model_name.lower() == 'maxout':
        model = rmodel.Maxout(num_classes=config['num_classes'])
    elif model_name.lower() == 'embedding':
        model = rmodel.Embedding(num_classes=config['num_classes'])
    elif model_name.lower() == 'vninet':
        model = rmodel.VniNet(num_classes=config['num_classes'])
    elif model_name.lower() == 'resnet18':
        model = rmodel.Resnet18(num_classes=config['num_classes'],
                                pretrained=config['pretrained'])
    elif model_name.lower() == 'vgg11bn':
        model = rmodel.VGG11BN(num_classes=config['num_classes'],
                               pretrained=config['pretrained'])
    else:
        raise ValueError('Unsupported model: ' + model_name)

    # criterion
    criterion = PredictLoss(pred_loss=config['pred_loss'])
    model = RecognitionModel(model, criterion)

    return model


def set_q_model(config):
    if 'q_model_name' in config:
        model_name = config['q_model_name']
    else:
        model_name = config['model_name'].split('_')[0]

    # model
    if model_name.lower() == 'resunet':
        model = qmodel.ResUnet()
    elif model_name.lower() == 'unet':
        model = qmodel.Unet()
    else:
        raise ValueError('Unsupported model: ' + model_name)

    # criterion
    criterion = IQALoss(pred_loss=config['pred_loss'], alpha=config['alpha'])
    model = IQAnModel(model, criterion)

    return model


def set_eye_dataloaders(config, mode='rtrain', pdfs=None):
    train_data = EyeDataset(dataset=config['dataset'],
                            mode=mode,
                            less_data=config['less_data'],
                            dfs=pdfs,
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


def set_face_dataloaders(config, mode='qtrain', dfs=[None, None]):

    train_data = FaceDataset(
        dataset=config['dataset'],
        mode=mode,
        less_data=config['less_data'],
        dfs=dfs[0],
    )
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data = FaceDataset(
        dataset=config['dataset'],
        mode='val',
        less_data=config['less_data'],
        dfs=dfs[1],
    )
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    return (train_data_loader, val_data_loader)