import os
import torch

from src.util import LoadConfig
from run.rtest import test as rtest


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.log_name = "test_set"

        self.r_cp_path = "checkpoints/0520_221932_r_maxout_/400_8.3929e-02.pth"
        self.visible = True
        self.debug = False
        self.less_data = False
        self.warmup = False

        self.batchsize = 16
        self.device = [2]
        self.num_workers = 0
        self.seed = 2358

        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        self.log_name = "test_{}_{}".format(
            os.path.basename(os.path.dirname(self.r_cp_path)),
            self.log_name,
        )
        # if self.debug or self.less_data != False:
        #     self.warmup = False


if __name__ == "__main__":
    # set config
    config = Config()

    # load checkpoint
    checkpoint = torch.load(config['r_cp_path'],
                            map_location=torch.device('cpu'))

    if checkpoint['cfg']['train_type'] == 'r':
        rtest(config, checkpoint)