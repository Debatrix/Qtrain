# coding=utf-8
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.util import *

tqdm_bar = '{desc} {n_fmt}/{total_fmt}-{percentage:3.0f}%|{rate_fmt}'


def train_body(config,
               model,
               dataloaders,
               optimizers,
               evaluation=None,
               log_writer=None,
               val_plot=None):
    tag = '' if 'tag' not in config else config['tag']
    # data
    train_data_loader, val_data_loader = dataloaders

    # optimizer and scheduler
    optimizer, scheduler = optimizers

    # Start!
    cur_epoch = 0 if 'cur_epoch' not in config else config['cur_epoch']
    error_num = 0
    print("{}[{}] {} Start training!".format(tag, cur_epoch, get_timestamp()))
    while cur_epoch < config['max_epochs']:
        cur_epoch += 1
        # runtime checkpoints
        model_checkpoint = model.state_dict()
        optimizer_checkpoint = optimizer.state_dict()
        # train
        model.init_train()
        model.train()
        # epoch_loss = 0
        epoch_loss = {}
        try:
            for train_data in tqdm(train_data_loader,
                                   desc='{}[{}] Epoch'.format(tag, cur_epoch),
                                   bar_format=tqdm_bar):

                loss = 0
                optimizer.zero_grad()
                loss = model.train_epoch(train_data)
                # epoch_loss += loss['loss'].item()
                for k, v in loss.items():
                    if k in epoch_loss:
                        epoch_loss[k] += v.cpu().item()
                    else:
                        epoch_loss[k] = v.cpu().item()
                loss['loss'].backward()
                optimizer.step()
                # Don't waste time
                assert not torch.isnan(loss['loss']), 'Nan!'
            model.train_finish()
        except AssertionError as err:
            error_num += 1
            model.load_state_dict(model_checkpoint)
            optimizer.load_state_dict(optimizer_checkpoint)
            err_info = '{}[{}] {} {}, back to last epoch'.format(
                tag, cur_epoch, get_timestamp(), err)
            print(err_info)
            if config['save_interval'] > 0:
                with open(os.path.join(config['cp_dir_path'], 'output.log'),
                          'a') as f:
                    f.write(err_info)
            cur_epoch -= 1
            assert error_num < config['log_interval']
            continue
        error_num = 0

        train_loss = epoch_loss['loss'] / len(train_data_loader.dataset)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()

        pinfo = "{}[{}] {} Training - loss: {:.2e}".format(
            tag, cur_epoch, get_timestamp(), train_loss)
        if len(epoch_loss) > 2:
            for k, v in epoch_loss.items():
                if k == 'loss':
                    continue
                else:
                    pinfo += ' {}: {:.2e}'.format(
                        k, v / len(train_data_loader.dataset))

        print(pinfo)
        if config['save_interval'] > 0:
            with open(os.path.join(config['cp_dir_path'], 'output.log'),
                      'a') as f:
                f.write(pinfo)
        if config['visible']:
            cur_lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Train/lr', cur_lr, cur_epoch)
            if len(epoch_loss) > 2:
                for k, v in epoch_loss.items():
                    log_writer.add_scalar('Train/' + k.capitalize(),
                                          v / len(train_data_loader.dataset),
                                          cur_epoch)
            else:
                log_writer.add_scalar('Train/Loss', train_loss, cur_epoch)

        # val
        val_save = None
        if cur_epoch % config['log_interval'] == 0 or config['debug']:
            # torch.cuda.empty_cache()
            model.eval()
            model.init_val()
            with torch.no_grad():
                for val_data in tqdm(val_data_loader,
                                     desc='{}[{}] Val'.format(tag, cur_epoch),
                                     bar_format=tqdm_bar):
                    val_save = model.val_epoch(val_data)

            val_result, val_save = evaluation(val_save,
                                              len(val_data_loader.dataset))

            val_info = "{}[{}] {} Val -".format(tag, cur_epoch,
                                                get_timestamp())
            for k, v in val_result.items():
                if 'loss' in k.lower():
                    val_info += " {}: {:.4e},".format(k, v)
                else:
                    val_info += " {}: {:.4f},".format(k, v)
            print(val_info)

            if config['save_interval'] > 0:
                with open(os.path.join(config['cp_dir_path'], 'output.log'),
                          'a') as f:
                    val_info += '\n'
                    f.write(val_info)
            if config['visible']:
                for k, v in val_result.items():
                    log_writer.add_scalar('Val/{}'.format(k), v, cur_epoch)
                if val_plot is not None:
                    val_plot(log_writer, cur_epoch, val_save)

        # checkpoint save
        if config['save_interval'] > 0 and cur_epoch % config[
                'save_interval'] == 0:
            save_path = os.path.join(
                config['cp_dir_path'],
                '{}_{:.4e}.pth'.format(cur_epoch, train_loss))
            model.save_checkpoint(save_path, info=config)
        elif config['save_interval'] == -1:
            save_path = os.path.join(config['cp_dir_path'],
                                     '{}.pth'.format(config['log_name']))
            model.save_checkpoint(save_path, info=config)
    return model


def base_train_head(config, model):
    # configure train
    print(config)
    set_random_seed(config['seed'])

    # configure model
    cp_config = model.load_checkpoint(config['cp_path'])
    model.to_device(config['device'])

    # checkpoint
    cp_dir_path = ''
    if config['save_interval'] > 0 or config['save_interval'] == -1:
        cp_dir_path = os.path.normcase(
            os.path.join('checkpoints', config['log_name']))
        os.makedirs(cp_dir_path)
        with open(os.path.join(cp_dir_path, 'output.log'), 'a') as f:
            if config['cp_path']:
                info = str(config) + '#' * 30 + '\npre_config:\n' + str(
                    cp_config) + '#' * 30 + '\n'
            else:
                info = str(config) + '#' * 30 + '\n'
            f.write(info)
        if len(config['git_hash']) > 0:
            os.system('git archive -o {} HEAD'.format(
                os.path.join(cp_dir_path, 'code.zip')))
    config['cp_dir_path'] = cp_dir_path

    log_writer = None
    if config['visible']:
        log_writer = SummaryWriter(os.path.join("log", config['log_name']))
        log_writer.add_text('cur_config', str(config))
        if config['cp_path']:
            log_writer.add_text('pre_config', cp_config.__str__())
    return config, model, log_writer


def train(config,
          dataloaders,
          model,
          optimizers,
          evaluation=None,
          val_plot=None):
    config, model, log_writer = base_train_head(config, model)
    model = train_body(config, model, dataloaders, optimizers, evaluation,
                       log_writer, val_plot)
    return model
