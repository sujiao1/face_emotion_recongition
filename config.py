from module.drsnet import *
from utils.functions import *

""" Dataset parameters """


class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'dilated_se_resnet50_reoutdgs128_K_data'  # define your model
        self.dataset = 'CK+'
        self.root = 'models'
        self.output_stride = 8
        self.down_sample_rate = 32  # classic down sample rate, DO NOT CHANGE!
        self.se_mode = 2  # Squeeze and Excitation mode, 0->cSE, 1-> sSE, 2->scSE
        self.HDC = True  # Hybrid Dilated Convolution, type bool
        self.multi_grid = None  # multi_grid in DeepLabv3, type bool

        # dataset parameters
        self.rescale_size = 224  # rescale image when training
        self.image_size = 224  # the final image size after crop
        self.num_class = 8  # 8 classes for training
        self.dataset_root = 'datasets/img_label_data'
        self.test_data = 'datasets/CK+_imgs/test'
        self.dataloader_workers = 2
        self.shuffle = True  # if shuffle data when training
        self.train_batch = 16
        self.val_batch = 8
        self.test_batch = 1

        # train parameters
        self.num_epoch = 67
        self.base_lr = 0.0001
        # self.base_lr = 0.0001
        self.power = 0.9  # lr decay power
        self.head_lr_mult = 1
        self.backbone_lr_mult = 1
        self.momentum = 0.9
        # self.weight_decay = 0.0005
        self.weight_decay = 0.005
        self.should_val = True
        self.val_every = 1  # how often will we evaluate model on val set
        self.display = 1  # how often will we show train result
        self.kfold = 7

        # model restore parameters
        self.resume_from = 'models/models_dilated_se_resnet50_re_K_data_CK+/checkpoints/Checkpoint_best_epoch_65.pth.tar'  # None for train from scratch
        self.pre_trained_from = None  # None for train from scratch
        self.should_save = True
        self.save_every = 10  # how often will we save checkpoint

        # create training dir
        self.summary = True
        if self.summary:
            self.summary_dir, self.ckpt_dir = create_train_dir(self)


""" Class name transform """
# name2net = {'DRSNet': DRSNet}

def main():
    aa = Params()
    print_config(aa)
    print(aa.summary)
    print(aa.ckpt_dir)


if __name__ == '__main__':
    main()
