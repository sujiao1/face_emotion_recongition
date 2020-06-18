import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import torch
import cv2
import time
from multiprocessing import set_start_method


from config import *
# from datasets import *
from k_fold_dataset import *
from module.dilated_se_resnet import *
from module.se_resnet import *
from utils.functions import *
from utils.progressbar import bar
from src.detector import detect_faces
from mtcnn_landmarks_tailoring import *


class train_CK_net(nn.Module):
    def __init__(self, params, module=None):
        super(train_CK_net,self).__init__()
        # 传入config中定义的参数
        self.params = params
        print('params')
        print_config(params)

        # 创建数据集
        print('创建数据集并进行数据集转换')
        self.datasets = create_k_dataset(params)
        print('数据集创建成功')

        # 定义训练的epoch、loss等参数
        self.pb = bar()
        self.epoch = 0
        self.init_epoch = 0
        self.checkpoint_flag = False
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        self.best_val_acc = 0
        self.best_val_acc_epoch = 0
        # self.use_gpu = torch.cuda.is_available()
        # print('use_gpu:{}'.format(self.use_gpu))

        if self.params.summary:
            self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # 定义训练的网络
        print('构建初始化模型')
        self.model = dilated_se_Net50(params)
        # self.model = se_resnet_50(params)
        self.build_network(self.model)
        print('模型初始化完成')

        # 设置默认的损失函数
        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        # 设置优化器
        self.optim = torch.optim.Adam([{'params': self.model_params, 'lr_mult': self.params.backbone_lr_mult}],
                                      lr=self.params.base_lr,
                                      weight_decay=self.params.weight_decay)
        # 初始化数据集
        self.build_dataloader()

        # 加载数据
        self.load_checkpoint()
        self.load_model()

    # 前向传播
    def forward(self, x):
        logits = x
        logits = self.model(logits)
        return logits

    """
    进行训练及验证
    """
    def Train(self):
        """
        在n个epochs上进行训练
        """
        self.init_epoch = self.epoch

        if self.epoch >= self.params.num_epoch:
            print('Num_epoch should be smaller than current epoch. Skip training......\n')
        else:
            for _ in range(self.epoch, self.params.num_epoch):
                self.epoch += 1
                print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)

                # train one epoch
                self.train_one_epoch()

                # 展示train_loss
                if self.epoch % self.params.display == 0:
                    print('\tTrain loss: %.4f Train acc: %.4f' % (self.train_loss[-1], self.train_acc[-1]))

                # 保存
                if self.params.should_save:
                    if self.epoch % self.params.save_every == 0:
                        self.save_checkpoint()

                # 训练val epoch
                if self.params.should_val:
                    if self.epoch % self.params.val_every == 0:
                        self.val_one_epoch()
                        print('\tVal loss: %.4f Val acc: %.4f' % (self.val_loss[-1], self.val_acc[-1]))
                        if self.val_acc[-1] > self.best_val_acc:
                            self.best_val_acc = self.val_acc[-1]
                            self.best_val_acc_epoch = self.epoch
                            save_stat = {'epoch': self.epoch,
                                         'training_loss': self.train_loss,
                                         'val_loss': self.val_loss,
                                         'stat_dict': self.state_dict(),
                                         'optimizer': self.optim.state_dict(),
                                         'train_acc': self.train_acc,
                                         'best_val_acc': self.best_val_acc,
                                         'best_val_acc_epoch': self.best_val_acc_epoch}
                            torch.save(save_stat, self.params.ckpt_dir + 'Checkpoint_best_epoch_%d.pth.tar' % self.epoch)
                            print('best save.....')
                            print('best acc %f epoch %d' % (self.best_val_acc, self.best_val_acc_epoch))

                # 调整学习率
                self.adjust_lr()

            # 保存最终网络模型
            if self.params.should_save:
                self.save_checkpoint()

            # 可视化训练曲线
            self.plot_curve()

    def Test(self):
        """
        Test network on test set
        """
        print("Testing........")
        test_loss = 0
        total = 0
        correct = 0

        # set model val
        # torch.cuda.empty_cache()
        print(self.model)
        self.model.eval()


        # prepare test data
        test_size = len(self.datasets['test'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        # test for one epoch
        with torch.no_grad():
            for batch_idx, (img, label, label_name)in enumerate(self.test_loader):
                self.pb.click(batch_idx, total_batch)
                print('label')
                print(label)
                # image, label, name = batch['image'], batch['label'], batch['label_name']
                image_cuda, label_cuda = img.cuda(), label.cuda()
                # print(image_cuda.shape)
                print('labelcuda')
                print(label_cuda)
                # img_cuda = cv2.cvtColor(image_cuda, cv2.COLOR_BGR2RGB)
                # bounding_box, landmarks = detect_faces(image_cuda)

                label_name = label_name
                out = self(image_cuda)
                _, predicted = torch.max(out.data, 1)

                loss = self.loss(out, label_cuda.long())
                test_loss += loss.item()

                total += label_cuda.long().size(0)
                correct += predicted.eq(label_cuda.long().data).sum()
                test_acc = float(correct) / float(total)

                # 记录第一次损失
                if self.test_loss == [] and self.test_acc == []:
                    self.test_loss.append(test_loss)
                    self.test_acc.append(test_acc)
                    if self.params.summary:
                        self.summary_writer.add_scalar('loss/test_loss', test_loss, 0)
                        self.summary_writer.add_scalar('acc/test_acc', test_acc, 0)

        self.pb.close()
        test_loss /= total_batch
        self.test_loss.append(test_loss)
        # train_acc = correct/len(self.train_loader)
        print('test correct')
        print(correct)
        print('test total')
        print(total)
        test_acc = float(correct) / float(total)
        print('test acc')
        print(test_acc)
        self.test_acc.append(test_acc)

        # add to summary
        if self.params.summary:
            self.summary_writer.add_scalar('loss/test_loss', test_loss, self.epoch)
            self.summary_writer.add_scalar('acc/test_acc', test_acc, self.epoch)

        # for i in range(self.params.test_batch):
        #     idx = batch_idx * self.params.test_batch + i
        #     id_map = logits2trainId(out[i, ...])
        #     color_map = trainId2color(self.params.dataset_root, id_map, name=name[i])
        #     trainId2LabelId(self.params.dataset_root, id_map, name=name[i])
        #     image_orig = image[i].numpy().transpose(1, 2, 0)
        #     image_orig = image_orig * 255
        #     image_orig = image_orig.astype(np.uint8)
        #     if self.params.summary:
        #         self.summary_writer.add_image('test/img_%d/orig' % idx, image_orig, idx)
        #         self.summary_writer.add_image('test/img_%d/seg' % idx, color_map, idx)

    def train_one_epoch(self):
        total = 0
        num_segment = 0
        """
        针对每个epoch训练模型
        """
        print('Training')

        # 网络训练
        self.model.train()

        # 准备数据
        train_loss = 0
        correct = 0
        train_acc = 0

        train_size = len(self.datasets['train'])
        # print('train数据集大小')
        # print(train_size)
        # print('train数据集')
        # for imglabel in self.datasets['train']:
        #     print(imglabel)
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
            print('totalbatch')
            print(total_batch)

        else:
            total_batch = train_size // self.params.train_batch
            print(total_batch)

        # 针对数据进行训练
        # print('trainloader')
        # print(self.train_loader)

        # self.train_loader = pickle.dumps(self.train_loader, protocol=2)
        # self.train_loader = pickle.loads(self.train_loader)
        print(len(self.train_loader))
        print(self.train_loader.batch_size)
        # print(self.train_loader.__iter__())
        for batch_idx, (img, label, label_name) in enumerate(self.train_loader):
            self.pb.click(batch_idx, total_batch)
            # img, label = batch['img'], batch['label']
            # if self.use_gpu:
            img_cuda = img.cuda()
            # print('img_cuda')
            # print(img_cuda)
            # print(img_cuda.shape)
            label_cuda = label.cuda()
            # label_cuda = label_cuda.float()
            # print('label_cuda')
            # print(label_cuda)
            # print(label_cuda.shape)


            try:
                out = self(img_cuda)
                out = out.float()
                # time.sleep(0.003)
                # out = checkpoint_sequential(self.model, num_segment, img_cuda)
            except RuntimeError as exception:
                if "memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception


            # out = self(img_cuda)
            # out = out.float()
            # print('trainout')
            # print(out)
            # print(out.shape)

            # acc
            _, predicted = torch.max(out.data, 1)

            loss = self.loss(out, label_cuda.long())

            # l2 = torch.tensor(0.005).cuda()
            # lambda = torch.tensor(1.)
            l2_reg = torch.tensor(0.).cuda()
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.params.weight_decay * l2_reg

            # optimize
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # train loss计算
            train_loss += loss.item()

            # acc计算
            # correct += torch.sum(predicted == label_cuda.long().data)
            # print('train correct')
            # print(correct)
            # print('self.train_loader')
            # print(len(self.train_loader))
            total += label_cuda.long().size(0)
            correct += predicted.eq(label_cuda.long().data).sum()
            train_acc = float(correct) / float(total)



            # 记录第一次损失
            if self.train_loss == [] and self.train_acc == []:
                self.train_loss.append(train_loss)
                self.train_acc.append(train_acc)
                if self.params.summary:
                    self.summary_writer.add_scalar('loss/train_loss', train_loss, 0)
                    self.summary_writer.add_scalar('acc/train_acc', train_acc, 0)

        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)
        # train_acc = correct/len(self.train_loader)
        print('train correct')
        print(correct)
        print('train total')
        print(total)
        train_acc = float(correct)/float(total)
        print('train acc')
        print(train_acc)
        self.train_acc.append(train_acc)

        # add to summary
        if self.params.summary:
            self.summary_writer.add_scalar('loss/train_loss', train_loss, self.epoch)
            self.summary_writer.add_scalar('acc/train_acc', train_acc, self.epoch)

    # 验证集单个epoch训练
    def val_one_epoch(self):
        """
        针对每个epoch训练模型
        """
        print('val-Training')

        # 网络训练
        self.model.eval()

        # 准备数据
        val_loss = 0
        total = 0
        correct = 0
        val_acc = 0
        num_segments = 0

        val_size = len(self.datasets['val'])
        if val_size % self.params.train_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch

        # 针对数据进行训练
        for batch_idx, (img, label, label_name) in enumerate(self.val_loader):
            self.pb.click(batch_idx, total_batch)
            # img, label = batch['img'], batch['label']
            img_cuda = img.cuda()
            label_cuda = label.cuda()

            """
            try:
                out = self(img_cuda)
                out = out.float()
                # out = checkpoint_sequential(self.model, num_segments, img_cuda)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            """

            out = self(img_cuda)
            out = out.float()
            print('valout')
            print(out)
            # print(out.shape)

            # acc
            _, predicted = torch.max(out.data, 1)

            loss = self.loss(out, label_cuda.long())
            l2_reg = torch.tensor(0.).cuda()
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.params.weight_decay * l2_reg

            # optimize
            # self.optim.zero_grad()
            # loss.backward()
            # self.optim.step()

            # train loss计算
            val_loss += loss.item()

            # acc计算
            # correct += torch.sum(predicted == label_cuda.long().data)
            # print('val correct')
            # print(correct)
            # print('self.val_loader')
            # print(len(self.val_loader))
            total += label_cuda.long().size(0)
            correct += predicted.eq(label_cuda.long().data).sum()
            val_acc = float(correct) / float(total)

            # 记录第一次损失
            if self.val_loss == [] and self.val_acc == []:
                self.val_loss.append(val_loss)
                self.val_acc.append(val_acc)
                if self.params.summary:
                    self.summary_writer.add_scalar('loss/val_loss', val_loss, 0)
                    self.summary_writer.add_scalar('acc/val_acc', val_acc, 0)


        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)
        print('val correct')
        print(correct)
        print('val total')
        print(total)
        # val_acc = correct/len(self.val_loader)
        val_acc = float(correct)/float(total)
        print('val acc')
        print(val_acc)
        self.val_acc.append(val_acc)

        # add to summary
        if self.params.summary:
            self.summary_writer.add_scalar('loss/val_loss', val_loss, self.epoch)
            self.summary_writer.add_scalar('acc/val_acc', val_acc, self.epoch)


    # 调整学习率
    def adjust_lr(self):
        learing_rate = self.params.base_lr * (1-float(self.epoch)/self.params.num_epoch) ** self.params.power
        # ?????
        for param_group in self.optim.param_groups:
            param_group['lr'] = learing_rate
        print('Change learning rate into %f' % (learing_rate))
        if self.params.summary:
            self.summary_writer.add_scalar('learning rate', learing_rate, self.epoch)

    # 可视化训练曲线
    def plot_curve(self):
        """
                Plot train/val loss curve
                """
        x1 = np.arange(0, self.params.num_epoch + 1, dtype=np.int)
        x2 = np.linspace(0, self.epoch,
                         num=(self.epoch - 0) // self.params.val_every + 1, dtype=np.int64)
        # x3 = np.arange(self.init_epoch, self.params.num_epoch + 1, dtype=np.int)
        # x4 = np.linspace(self.init_epoch, self.epoch,
        #                  num=(self.epoch - self.init_epoch) // self.params.val_every + 1, dtype=np.int64)
        # plt.subplot(2, 1, 1)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.subplot(2, 1, 2)
        # plt.plot(x1, self.train_acc, label='train_acc')
        # plt.plot(x2, self.val_acc, label='val_acc')
        # plt.legend(loc='best')
        # plt.title('Train/Val acc')
        # plt.grid()
        # plt.xlabel('Epoch')
        # plt.ylabel('acc')
        # plt.show()
        plt.savefig('resultimg/CK+_re_k_data_loss_0_66.jpg')
        # plt.savefig('resultimg/CK+_re_k_data_acc_0_66.jpg')

    # 保存模型
    def save_checkpoint(self):
        save_dict = {'epoch': self.epoch,
                     'training_loss': self.train_loss,
                     'val_loss': self.val_loss,
                     'stat_dict': self.state_dict(),
                     'optimizer': self.optim.state_dict(),
                     'train_acc': self.train_acc,
                     'val_acc': self.val_acc}
                     # 'best_val_acc': self.best_val_acc,
                     # 'best_val_acc_epoch': self.best_val_acc_epoch
        torch.save(save_dict, self.params.ckpt_dir + 'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        print('Checkpoint saved')

    def load_model(self):
        """
        加载预训练模型，没有训练好的模型时进行加载
        """
        if self.checkpoint_flag:
            print('Skip Loading Pre-trained Model......')
        else:
            if self.params.pre_trained_from is not None and os.path.exists((self.params.pre_trained_from)):
                # print('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                # pretrain = torch.load(self.params.pre_trained_from)
                # self.load_state_dict(pretrain)
                # print('Pre-trained Model Loaded!\n')
                try:
                    print('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                    pretrain = torch.load(self.params.pre_trained_from)
                    self.load_state_dict(pretrain)
                    print('Pre-trained Model Loaded!\n')
                except:
                    print('Cannot load pre-trained model. Start training......\n')
            else:
                print('Pre-trained model do not exits. Start training......\n')

    def load_checkpoint(self):
        # 从给定路径中加载checkpoint
        if self.params.resume_from is not None and os.path.exists(self.params.resume_from):
            # print('loading checkpoint at %s' % self.params.resume_from)
            # ckpt = torch.load(self.params.resume_from)
            # model_dict = ckpt['stat_dict']
            # print(model_dict)
            # dicts = list(model_dict)
            # print('dicts')
            # for k, v in enumerate(dicts):
            #     print(k, v)
            # self.epoch = ckpt['epoch']
            # # print(self.epoch)
            # self.train_loss = ckpt['training_loss']
            # self.val_loss = ckpt['val_loss']
            # print(self.train_loss)
            # print(self.val_loss)
            # self.load_state_dict(ckpt['stat_dict'])
            # # print('statedict')
            # # print(self.load_state_dict(ckpt['stat_dict']))
            # dict_name = list(self.load_state_dict(ckpt['stat_dict']))
            # for i, p in enumerate(dict_name):
            #     print(i, p)
            # self.optim.load_state_dict(ckpt['optimizer'])
            # # print('optimizer')
            # # print(ckpt['optimizer'])
            # print('checkpoint loaded')
            # print('Current Epoch: %d' % self.epoch)
            try:
                print('loading checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                self.train_loss = ckpt['training_loss']
                self.val_loss = ckpt['val_loss']
                # self.train_acc = ckpt['train_acc']
                # self.val_acc = ckpt['val_acc']
                # print(self.train_loss)
                # print(self.val_loss)
                self.load_state_dict(ckpt['stat_dict'])
                print('statedict')
                print(self.load_state_dict(ckpt['stat_dict']))
                self.optim.load_state_dict(ckpt['optimizer'])
                print('optimizer')
                print(self.optim.load_state_dict(ckpt['optimizer']))
                print('checkpoint loaded')
                print('Current Epoch: %d' % self.epoch)
                print('准确率')
                print(ckpt['best_val_acc'])
                try:
                    self.train_loss = ckpt['training_loss']
                    self.val_loss = ckpt['val_loss']
                    # self.best_val_acc = ckpt['best_val_acc']
                    # self.best_val_acc_epoch = ckpt['best_val_acc_epoch']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.load_state_dict(ckpt['stat_dict'])
                print('stat_dict')
                print(self.load_state_dict(ckpt['stat_dict']))
                self.optim.load_state_dict(ckpt['optimizer'])
                # print('optimizer')
                # print(self.optim.load_state_dict(ckpt['optimizer']))
                print('checkpoint loaded')
                print('Current Epoch: %d' % self.epoch)
                self.checkpoint_flag = True
            except:
                print('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.resume_from)
        else:
            print('Checkpoint do not exists. Start loading pre-trained model......')

    def build_dataloader(self):
        self.train_loader = data.DataLoader(self.datasets['train'],
                                            batch_size=self.params.train_batch,
                                            shuffle=self.params.shuffle,
                                            num_workers=self.params.dataloader_workers,
                                            pin_memory=False)
        self.val_loader = data.DataLoader(self.datasets['val'],
                                          batch_size=self.params.val_batch,
                                          shuffle=self.params.shuffle,
                                          num_workers=self.params.dataloader_workers,
                                          pin_memory=False)
        self.test_loader = data.DataLoader(self.datasets['test'],
                                           batch_size=self.params.test_batch,
                                           shuffle=False)

    # 模型构建
    def build_network(self, model=None):
        if model is None:
            if self.model is None:
                self.model = DRSNet(self.params)
        else:
            self.model = model

        self.cuda()
        self.initialize()

    def initialize(self):
        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.model_params = []
        # for m in self.model:
        self.model_params.extend(list(self.model.parameters()))


def main():
    # set_start_method('spawn')
    params = Params()
    params.dataset_root = 'datasets/img_label_data'
    print('得到h5文件下的数据集')
    # datasets = create_dataset(params)
    trainCK = train_CK_net(params)
    # trainCK.Train()
    trainCK.Test()


if __name__ == '__main__':
    main()