from src.util import LoadConfig
from run.rtrain import train as rtrain
from run.qtrain import train as qtrain
from run.rqtrain import train as rqtrain


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "LG4000"
        self.log_name = "LG4000"

        self.dataset = 'LG4000'
        self.q_cp_path = ""
        self.r_cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = -1
        self.less_data = False
        self.debug = False

        self.train_type = 'rq'
        self.q_model_name = 'Unet'
        self.r_model_name = 'maxout'
        self.pretrained = True

        self.rec_loss = 'ce'
        self.qua_loss = 'mse'
        self.weight = 'gaussian'
        self.alpha = 0.5

        self.q_batchsize = 8
        self.r_batchsize = 32
        self.device = [3, 2, 0, 1]
        self.num_workers = 4
        self.seed = 2358

        self.max_epochs = 1500
        self.q_max_epoch = [14, 200]
        self.r_max_epoch = [15, 250]
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 0.01
        self._auto_setting()
        self.apply()

    def _auto_setting(self):
        if self.train_type == 'q':
            self.model_name = self.q_model_name
        elif self.train_type == 'r':
            self.model_name = self.r_model_name
        else:
            self.model_name = '{}_{}'.format(self.q_model_name,
                                             self.r_model_name)
        if self.debug:
            self.q_max_epoch = [2, 2]
            self.r_max_epoch = [2, 2]

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
        self.num_classes = -1


if __name__ == "__main__":
    # set config
    config = Config()
    config = config.__dict__

    if config['train_type'] == 'r':
        rtrain(config)
    elif config['train_type'] == 'q':
        qtrain(config)
    elif config['train_type'] == 'rq':
        rqtrain(config)
