from src import qmodel, rmodel
from src.loss import IQALoss, PredictLoss
from src.framework import IQAnModel, RecognitionModel


def set_r_model(config):
    if 'r_model_name' in config:
        model_name = config['r_model_name']
    else:
        model_name = config['model_name'].split('_')[-1]

    # model
    if model_name.lower() == 'maxout':
        model = rmodel.Maxout(num_classes=config['num_classes'])
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