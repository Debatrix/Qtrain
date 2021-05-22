from src.util import LoadConfig
from run.rtrain import train as rtrain
from run.qtrain import train as qtrain


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "sl1_sgd"
        self.log_name = "sl1_sgd"

        self.dataset = 'distance'
        self.r_cp_path = "checkpoints/0520_221932_r_maxout_/375_8.9186e-02.pth"
        self.q_cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = -1
        self.less_data = False
        self.debug = False

        self.train_type = 'q'
        self.model_name = 'Unet'
        self.pretrained = True

        self.pred_loss = 'sl1'
        self.alpha = 0.5

        self.batchsize = 16
        self.device = [0, 1]
        self.num_workers = 4
        self.seed = 2358

        self.max_epochs = 2500
        self.lr = 2e-4
        self.momentum = 0.9
        self.weight_decay = 1e-2

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
        self.num_classes = None


if __name__ == "__main__":
    # set config
    config = Config()

    if config['train_type'] == 'r':
        rtrain(config)
    elif config['train_type'] == 'q':
        qtrain(config)
