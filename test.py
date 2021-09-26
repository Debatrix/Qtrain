import os
import torch
import copy
from glob import glob
import shutil

from src.util import LoadConfig, get_datestamp
from run.rtest import test as rtest


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.log_name = ""
        self.log_info = ""

        self.r_cp_path = "checkpoints/0924_050232_r_maxout_more_dataset_more_round/0924_050232_r_maxout_more_dataset_more_round.pth"
        self.dataset = ['distance', 'thousand']
        self.visible = True
        self.debug = False
        self.less_data = False
        self.warmup = False

        self.batchsize = 32
        self.device = [0, 1, 2]
        self.num_workers = 2
        self.seed = 0

        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        pass
        # self.log_name = "{}_{}".format(
        #     os.path.basename(self.r_cp_path).split('.')[0],
        #     self.log_name,
        # )
        # if self.debug or self.less_data != False:
        #     self.warmup = False


if __name__ == "__main__":
    # set config
    config = Config()
    config = config.__dict__

    # load checkpoint
    if os.path.isfile(config['r_cp_path']):
        checkpoint = torch.load(config['r_cp_path'],
                                map_location=torch.device('cpu'))
        config['log_name'] = "{}_{}".format(
            config['log_name'],
            '_'.join(os.path.dirname(config['r_cp_path']).split('_')[:2]),
        )

        datasets = config['dataset'] if isinstance(
            config['dataset'], (list, tuple)) else [config['dataset']]
        for dataset in datasets:
            print('\ndataset: ', dataset)
            config['dataset'] = dataset
            config['log_name'] = "{}_{}".format(
                config['log_name'],
                dataset,
            )
            rtest(config, checkpoint)
    else:
        # clean
        for x in glob('log/test/*'):
            shutil.rmtree(x)
        cp_list = glob(os.path.join(config['r_cp_path'], '*.pth'))
        for idx, path in enumerate(cp_list):
            cur_config = copy.deepcopy(config)
            config['r_cp_path'] = path

            config['dataset'] = 'distance'
            config['log_name'] = "{}_{}_{}".format(
                get_datestamp(),
                os.path.basename(path).replace('.', '_'),
                cur_config['dataset'],
            )

            print('\n{}/{}: {}'.format(2 * idx + 0, 2 * len(cp_list),
                                       cur_config['log_name']))

            checkpoint = torch.load(cur_config['r_cp_path'],
                                    map_location=torch.device('cpu'))
            rtest(cur_config, checkpoint)

            cur_config.dataset = 'thousand'
            config['log_name'] = "{}_{}_{}".format(
                get_datestamp(),
                os.path.basename(path).replace('.', '_'),
                cur_config['dataset'],
            )
            print('\n{}/{}: {}'.format(2 * idx + 1, 2 * len(cp_list),
                                       cur_config['log_name']))

            checkpoint = torch.load(cur_config['r_cp_path'],
                                    map_location=torch.device('cpu'))
            rtest(cur_config, checkpoint)
