import torch
from torch.utils.data import DataLoader

from src import qmodel, rmodel
from src.loss import IQALoss, PredictLoss
from src.framework import IQAnModel, RecognitionModel, TripRecognitionModel
from src.dataset import get_eye_dataset, EyePairDataset, FaceDataset


def set_r_model(config):
    if 'r_model_name' in config:
        model_name = config['r_model_name']
    else:
        model_name = config['model_name'].split('_')[-1]

    # model
    if model_name.lower() == 'maxout':
        model = rmodel.Maxout(num_classes=config['num_classes'])
    elif model_name.lower() == 'maxout_o':
        model = rmodel.MaxoutO(num_classes=config['num_classes'])
    elif model_name.lower() == 'nlightcnn':
        model = rmodel.LightCNN(num_classes=config['num_classes'], norm=True)
    elif model_name.lower() == 'lightcnn':
        model = rmodel.LightCNN(num_classes=config['num_classes'], norm=False)
    elif model_name.lower() == 'embedding':
        model = rmodel.Embedding(num_classes=config['num_classes'])
    elif model_name.lower() == 'vninet':
        model = rmodel.VniNet(num_classes=config['num_classes'])
    elif model_name.lower() == 'resnet18':
        model = rmodel.Resnet18(num_classes=config['num_classes'],
                                norm=False,
                                pretrained=config['pretrained'])
    elif model_name.lower() == 'resnet18n':
        model = rmodel.Resnet18(num_classes=config['num_classes'],
                                norm=True,
                                pretrained=config['pretrained'])
    elif model_name.lower() == 'vgg11bn':
        model = rmodel.VGG11BN(num_classes=config['num_classes'],
                               pretrained=config['pretrained'])
    else:
        raise ValueError('Unsupported model: ' + model_name)

    # criterion
    criterion = PredictLoss(loss_type=config['rec_loss'])
    if config['rec_loss'] == 'triplet':
        model = TripRecognitionModel(model, criterion)
    else:
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
    criterion = IQALoss(loss_type=config['qua_loss'], alpha=config['alpha'])
    model = IQAnModel(model, criterion)

    return model


def set_eye_dataloaders(config, mode='rtrain', pdfs=None):
    if config['qua_loss'] == 'triplet':
        train_data = EyePairDataset(dataset=config['dataset'],
                                    mode=mode,
                                    less_data=config['less_data'],
                                    dfs=pdfs,
                                    weight=config['weight'])
    else:
        train_data, num_classes = get_eye_dataset(
            datasets=config['dataset'],
            mode=mode,
            less_data=config['less_data'],
            dfs=pdfs,
            weight=config['weight'])
    train_data_loader = DataLoader(train_data,
                                   config['r_batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data, num_classes = get_eye_dataset(datasets=config['dataset'],
                                            mode='val',
                                            less_data=config['less_data'])
    val_data_loader = DataLoader(val_data,
                                 config['r_batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    # num_classes = train_data.num_classes
    return (train_data_loader, val_data_loader), num_classes


def set_face_dataloaders(config, mode='qtrain', dfs=[None, None]):

    train_data = FaceDataset(
        dataset=config['dataset'],
        mode=mode,
        less_data=config['less_data'],
        dfs=dfs[0],
    )
    train_data_loader = DataLoader(train_data,
                                   config['q_batchsize'],
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
                                 config['q_batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    return (train_data_loader, val_data_loader)