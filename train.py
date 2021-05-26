from src.util import LoadConfig
from run.rtrain import train as rtrain
from run.qtrain import train as qtrain
from run.rqtrain import train as rqtrain


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "vninet"
        self.log_name = "vninet"

        self.dataset = 'distance'
        self.q_cp_path = ""
        self.r_cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = 25
        self.less_data = False
        self.debug = False

        self.train_type = 'r'
        self.q_model_name = 'Unet'
        self.r_model_name = 'vninet'
        self.pretrained = True

        self.pred_loss = 'sl1'
        self.weight = 'gaussian'
        self.alpha = 0.1

        self.batchsize = 16
        self.device = [2, 3]
        self.num_workers = 0
        self.seed = 2358

        self.max_epochs = 500
        self.q_max_epoch = [14, 150]
        self.r_max_epoch = [15, 200]
        self.lr = 4e-3
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
        # if self.debug:
        #     self.q_max_epoch = [2, 2]
        #     self.r_max_epoch = [2, 2]

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
