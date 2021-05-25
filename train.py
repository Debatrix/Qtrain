from src.util import LoadConfig
from run.rtrain import train as rtrain
from run.qtrain import train as qtrain


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = "qrtrain"
        self.log_name = "qrtrain"

        self.dataset = 'distance'
        self.q_cp_path = "checkpoints/0522_223707_q_Unet_sl1_adam/500_8.4497e-03.pth"
        self.r_cp_path = ""
        self.visible = True
        self.log_interval = 10
        self.save_interval = -1
        self.less_data = False
        self.debug = False

        self.train_type = 'r'
        self.q_model_name = 'ResUnet'
        self.r_model_name = 'maxout'
        self.pretrained = True

        self.q_pred_loss = 'sl1'
        self.weight = 'gaussian'
        self.alpha = 0.5

        self.batchsize = 8
        self.device = [0, 1]
        self.num_workers = 2
        self.seed = 2358

        self.max_epochs = 300
        self.lr = 2e-3
        self.momentum = 0.9
        self.weight_decay = 1e-2

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
