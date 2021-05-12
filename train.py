import torch
from torch.utils.data import DataLoader

from src.base_train import train
from src.util import LoadConfig
from src.dataset import BaseDataset, PairedDataset
from src.model import *
from src.framework import PredictModel, SiameseModel
from src.loss import PredictLoss, SiameseLoss
from src.evaluation import evaluation, val_plot


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "newdata"
        self.log_name = ""

        self.dataset = '2104'
        self.cp_path = ""
        self.visible = True
        self.log_interval = 5
        self.save_interval = 5
        self.less_data = 0.01
        self.debug = True

        self.bg_ths = 9
        self.bg_classifier = 'none'  # none, integration, independence
        self.resize = (448, 224)  # (crop,resize)

        self.model_name = 'resnet18'
        self.pretrained = True

        self.train_type = 'single'  # single, siamese
        self.pred_loss = 'ce'
        self.feat_loss = 'cos'
        self.margin = 0.5
        self.alpha = 4

        self.batchsize = 32
        self.device = [2, 3]
        self.num_workers = 8
        self.seed = 2358

        self.max_epochs = 150
        self.lr = 2e-3
        self.momentum = 0.9
        self.weight_decay = 0.1

        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        self.log_name = "{}_{}_{}".format(
            self.train_type,
            self.model_name,
            self.log_name,
        )
        self.info = "{}, {}, {}".format(
            self.train_type,
            self.model_name,
            self.info,
        )


def set_dataloaders(config):
    if config['train_type'] == 'siamese':
        train_data = PairedDataset(dataset=config['dataset'],
                                   mode='train',
                                   less_data=config['less_data'],
                                   resize=config['resize'])
        train_data_loader = DataLoader(train_data,
                                       config['batchsize'],
                                       drop_last=True,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=config['num_workers'])
    else:
        train_data = BaseDataset(
            dataset=config['dataset'],
            mode='train',
            less_data=config['less_data'],
            resize=config['resize'],
        )
        train_data_loader = DataLoader(train_data,
                                       config['batchsize'],
                                       drop_last=True,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=config['num_workers'])
    val_data = BaseDataset(
        dataset=config['dataset'],
        mode='val',
        less_data=config['less_data'],
        resize=config['resize'],
    )
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    return train_data_loader, val_data_loader


def set_model(config):

    # model
    if config['model_name'].lower() == 'resnet18':
        model = Resnet18(bg_classifier=config['bg_classifier'],
                         pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'resnet18d':
        model = Resnet18(bg_classifier=config['bg_classifier'],
                         pretrained=config['pretrained'],
                         drop=True)
    elif config['model_name'].lower() == 'resnet34':
        model = Resnet34(bg_classifier=config['bg_classifier'],
                         pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'resnet50':
        model = Resnet50(bg_classifier=config['bg_classifier'],
                         pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'resnet101':
        model = Resnet101(bg_classifier=config['bg_classifier'],
                          pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'resnet101d':
        model = Resnet101(bg_classifier=config['bg_classifier'],
                          pretrained=config['pretrained'],
                          drop=True)
    elif config['model_name'].lower() == 'mobilenet':
        model = MobileNet(bg_classifier=config['bg_classifier'],
                          pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'inception':
        model = Inception(bg_classifier=config['bg_classifier'],
                          pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'vgg16bn':
        model = VGG16BN(bg_classifier=config['bg_classifier'],
                        pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'vgg11bn':
        model = VGG11BN(bg_classifier=config['bg_classifier'],
                        pretrained=config['pretrained'])
    elif config['model_name'].lower() == 'senet18':
        model = SEnet18(bg_classifier=config['bg_classifier'],
                        pretrained=config['pretrained'])

    else:
        raise ValueError('Unsupported model: ' + config['model_name'])

    # criterion
    if config['train_type'] == 'siamese':
        criterion = SiameseLoss(pred_loss=config['pred_loss'],
                                feat_loss=config['feat_loss'],
                                margin=config['margin'],
                                alpha=config['alpha'])
        model = SiameseModel(model, criterion)
    else:
        criterion = PredictLoss(pred_loss=config['pred_loss'])
        model = PredictModel(model, criterion)

    return model


if __name__ == "__main__":
    # set config
    config = Config()

    # data
    print('Loading Data')
    dataloaders = set_dataloaders(config)

    # model and
    model = set_model(config)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=config['lr'],
    #     momentum=0.9,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['max_epochs'] // 3,
    # )

    optimizers = (optimizer, scheduler)

    # train
    train(config, dataloaders, model, optimizers, evaluation, val_plot)
