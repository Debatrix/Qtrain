from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.util import *
from src.base_train import train_body
from src.dataset import EyeDataset, FaceDataset
from src.evaluation import q_val_plot, r_evaluation, r_val_plot, q_evaluation
from run.set_model import set_q_model, set_r_model, set_eye_dataloaders, set_face_dataloaders


def rtrain(config, model, pdfs, log_writer):

    # data
    print('Loading Iris Data')
    dataloaders, num_classes = set_eye_dataloaders(config, 'rtrain', pdfs)
    config['num_classes'] = num_classes

    # model
    if model is None:
        model = set_r_model(config)
        cp_config = model.load_checkpoint(config['r_cp_path'])
        model.to_device(config['device'])
        if log_writer and config['r_cp_path']:
            log_writer.add_text('pre_config', cp_config.__str__())

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

    optimizer = torch.optim.SGD(params,
                                lr=config['lr'],
                                momentum=0.9,
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=config['log_interval'] * 2,
        verbose=True)

    optimizers = (optimizer, scheduler)

    model = train_body(config, model, dataloaders, optimizers, r_evaluation,
                       log_writer, r_val_plot)
    return model


def qtrain(config, model, dfs, log_writer):
    # data
    print('Loading Face Data')
    dataloaders = set_face_dataloaders(config, 'qtrain', dfs)

    # model
    if model is None:
        model = set_q_model(config)
        cp_config = model.load_checkpoint(config['q_cp_path'])
        model.to_device(config['device'])
        if log_writer and config['q_cp_path']:
            log_writer.add_text('pre_config', cp_config.__str__())

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
    optimizers = (optimizer, scheduler)

    model = train_body(config, model, dataloaders, optimizers, q_evaluation,
                       log_writer, q_val_plot)
    return model


def generate_dfs(config, model):
    if model is None:
        dfs = None
    else:
        dfs = []
        data_loader, _ = set_eye_dataloaders(config, 'qtrain')
        train_data_loader, val_data_loader = data_loader
        # run
        print('Generating dfs')
        model.eval()
        model.init_val()
        with torch.no_grad():
            for test_data in train_data_loader:
                model.val_epoch(test_data)
        _, val_save = r_evaluation(
            model.val_save,
            len(train_data_loader.dataset),
        )
        dfs.append(val_save['dfs'])

        model.init_val()
        with torch.no_grad():
            for test_data in val_data_loader:
                model.val_epoch(test_data)
        _, val_save = r_evaluation(
            model.val_save,
            len(val_data_loader.dataset),
        )
        dfs.append(val_save['dfs'])

    return dfs


def generate_pdfs(config, model=None):
    if model is None:
        pdfs = None
    else:
        pdfs = []
        train_data_loader, _ = set_face_dataloaders(config, 'rtrain')

        # run
        print('Generating pdfs')
        model.eval()
        model.init_val()
        with torch.no_grad():
            for test_data in train_data_loader:
                model.val_epoch(test_data)
        _, val_save = q_evaluation(
            model.val_save,
            len(train_data_loader.dataset),
        )
        pdfs = val_save['pdfs']

    return pdfs


def train(config):
    # configure train
    print(config)
    set_random_seed(config['seed'])
    log_name = config['log_name']

    # checkpoint
    cp_dir_path = ''
    if config['save_interval'] > 0 or config['save_interval'] == -1:
        cp_dir_path = os.path.normcase(
            os.path.join('checkpoints', config['log_name']))
        os.makedirs(cp_dir_path)
        os.makedirs(os.path.join(cp_dir_path, 'rmodel'))
        os.makedirs(os.path.join(cp_dir_path, 'qmodel'))
        with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
            info = str(config) + '#' * 30 + '\n'
            f.write(info)
        if len(config['git_hash']) > 0:
            os.system('git archive -o {} HEAD'.format(
                os.path.join(cp_dir_path, 'code.zip')))

    r_log_writer, q_log_writer = None, None
    if config['visible']:
        r_log_writer = SummaryWriter(os.path.join("log", log_name + '_r'))
        r_log_writer.add_text('cur_config', str(config))
        q_log_writer = SummaryWriter(os.path.join("log", log_name + '_q'))
        q_log_writer.add_text('cur_config', str(config))

    # init

    r_model, q_model = None, None
    pdfs, dfs = None, None
    rround, qround = 0, 0

    while rround + qround < config['r_max_epoch'][0] + config['q_max_epoch'][0]:

        if rround < config['r_max_epoch'][0]:
            # run rtrain
            rround += 1
            config['tag'] = 'R{}'.format(rround)
            config['log_name'] = log_name + '_R'
            config['cur_epoch'] = (rround - 1) * config['r_max_epoch'][1]
            config['max_epochs'] = rround * config['r_max_epoch'][1]
            config['cp_dir_path'] = os.path.join(cp_dir_path, 'rmodel')
            pdfs = generate_pdfs(config, q_model)
            r_model = rtrain(config, r_model, pdfs, r_log_writer)
            save_path = os.path.join(config['cp_dir_path'],
                                     'Round{}.pth'.format(rround))
            r_model.save_checkpoint(save_path, info=config)

        if qround < config['q_max_epoch'][0]:
            # run qtrain
            qround += 1
            config['tag'] = 'Q{}'.format(qround)
            config['log_name'] = log_name + '_Q'
            config['cur_epoch'] = (qround - 1) * config['q_max_epoch'][1]
            config['max_epochs'] = qround * config['q_max_epoch'][1]
            config['cp_dir_path'] = os.path.join(cp_dir_path, 'qmodel')
            dfs = generate_dfs(config, r_model)
            q_model = qtrain(config, q_model, dfs, q_log_writer)
            save_path = os.path.join(config['cp_dir_path'],
                                     'Round{}.pth'.format(qround))
            q_model.save_checkpoint(save_path, info=config)
