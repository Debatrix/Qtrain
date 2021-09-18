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

        self.r_cp_path = "checkpoints/recognition/"
        self.dataset = None
        self.visible = True
        self.debug = False
        self.less_data = False
        self.warmup = False

        self.batchsize = 32
        self.device = [0, 1, 2, 3]
        self.num_workers = 0
        self.seed = 2358

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

    # load checkpoint
    if os.path.isfile(config['r_cp_path']):
        checkpoint = torch.load(config['r_cp_path'],
                                map_location=torch.device('cpu'))
        config.log_name = "{}_{}".format(
            os.path.basename(config['r_cp_path']),
            config['log_name'],
        )
        rtest(config, checkpoint)
    else:
        # clean
        for x in glob('log/test2/*'):
            shutil.rmtree(x)
        cp_list = glob(os.path.join(config['r_cp_path'], '*.pth'))
        for idx, path in enumerate(cp_list):
            cur_config = copy.deepcopy(config)
            cur_config.r_cp_path = path

            cur_config.dataset = 'distance'
            cur_config.log_name = "{}_{}_{}".format(
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
            cur_config.log_name = "{}_{}_{}".format(
                get_datestamp(),
                os.path.basename(path).replace('.', '_'),
                cur_config['dataset'],
            )
            print('\n{}/{}: {}'.format(2 * idx + 1, 2 * len(cp_list),
                                       cur_config['log_name']))

            checkpoint = torch.load(cur_config['r_cp_path'],
                                    map_location=torch.device('cpu'))
            rtest(cur_config, checkpoint)