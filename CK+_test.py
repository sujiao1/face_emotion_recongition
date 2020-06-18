import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image
from torchvision import transforms, datasets
from src.box_utils import _preprocess, preprocess_input

from config import *
# from datasets import *
from k_fold_dataset import *
from module.dilated_se_resnet import *
from module.se_resnet import *
from utils.functions import *
from utils.progressbar import bar
from src.detector import detect_faces
from mtcnn_landmarks_tailoring import *



label_dict = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'neutral',
    6: 'sadness',
    7: 'surprise'
}


class test_CK_net(nn.Module):
    def __init__(self, params, module=None):
        super(test_CK_net,self).__init__()
        # 传入config中定义的参数
        self.params = params
        print('params')
        print_config(params)
        self.test_loss = []
        self.test_acc = []
        # self.data_dir = data_dir


        # 定义训练的网络
        print('构建初始化模型')
        self.model = dilated_se_Net50(params)
        # self.model = se_resnet_50(params)
        self.build_network(self.model)
        print('模型初始化完成')

        # 设置默认的损失函数
        self.loss = nn.CrossEntropyLoss(ignore_index=255)

        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])])

        self.trans = transforms.Compose([
            transforms.Resize(512)
        ])


        # 设置优化器
        self.optim = torch.optim.Adam([{'params': self.model_params, 'lr_mult': self.params.backbone_lr_mult}],
                                      lr=self.params.base_lr,
                                      weight_decay=self.params.weight_decay)

        self.dataset = datasets.ImageFolder(self.params.test_data)
        self.dataset.idx_to_class = {k: v for v, k in self.dataset.class_to_idx.items()}
        self.test_loader = data.DataLoader(self.dataset,
                                           batch_size=self.params.test_batch,
                                           shuffle=False,
                                           collate_fn=lambda x: x[0])

        # 加载数据
        self.load_checkpoint()

    def Test(self):
        """
        Test network on test set
        """
        print("Testing........")
        test_loss = 0
        total = 0
        correct = 0
        test_data = []
        save_img = False
        i = 0
        labels = []
        imgs = []
        img_labels = []

        # set model val
        # torch.cuda.empty_cache()
        print(self.model)
        self.model.eval()

        # prepare test data
        # test_size = len(self.datasets['test'])
        # if test_size % self.params.test_batch != 0:
        #     total_batch = test_size // self.params.test_batch + 1
        # else:
        #     total_batch = test_size // self.params.test_batch

        # test for one epoch
        for img, idx in self.test_loader:
            # print(expression)
            # print(img)
            # if expression == 0:
            name = self.dataset.idx_to_class[idx]
            label = self.dataset.class_to_idx[name]
            # print('label')
            # print(label)
            # label_array = np.asarray(label)
            # print('label_array')
            # print(label_array)
            label = torch.from_numpy(np.asarray([label])).float()
            print('label_array')
            print(label)
            # label_loader = data.dataloader(label)
            label_cuda = label.cuda()
            print('labelcuda')
            print(label_cuda)
            # print(self.dataset.class_to_idx[name])
            print(name)
            print('img类型')
            print(type(img))
            # image = transforms.ToPILImage()(img).convert('RGB')

            # img = Image.open(img_path)
            print(img.size)

            if len(img.size) == 2:
                image = img.convert('RGB')
            else:
                image = img

            i += 1

            h, w = image.size  # 图像的宽度和高度
            print(image.size)
            print(type(image))

            # print(img.dtype)
            bounding_box, landmarks = detect_faces(image)
            print('landmarks')
            print(landmarks)
            print(bounding_box.shape)
            # faces_count = landmarks.shape[0]
            # if faces_count > 1:
            #     print('图片中有多个人脸')
            #     return
            # elx, erx, nx, mlx, mrx, ely, ery, ny, mly, mry = landmarks[0]
            #
            # # 计算旋转角度
            # angle = calculate_angle(elx, ely, erx, ery)
            #
            # # 旋转图像
            # img_rotated = img.rotate(angle, expand=1)
            # ww, hh = img_rotated.size  # 旋转后图像的宽度和高度
            #
            # # 对齐后的位置
            # elx, ely = pos_transform(angle, elx, ely, w, h)
            # erx, ery = pos_transform(angle, erx, ery, w, h)
            # nx, ny = pos_transform(angle, nx, ny, w, h)
            # mlx, mly = pos_transform(angle, mlx, mly, w, h)
            # mrx, mry = pos_transform(angle, mrx, mry, w, h)
            #
            # # 基本参数
            # eye_width = erx - elx  # 两眼之间的距离
            # ecx, ecy = (elx + erx) / 2.0, (ely + ery) / 2.0  # 两眼中心坐标
            # mouth_width = mrx - mlx  # 嘴巴的宽度
            # mcx, mcy = (mlx + mrx) / 2.0, (mly + mry) / 2.0  # 嘴巴中心坐标
            # em_height = mcy - ecy  # 两眼睛中心到嘴巴中心高度
            # fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # 人脸中心坐标
            #
            # # 纯脸
            # if eye_width > em_height:
            #     alpha = eye_width
            # else:
            #     alpha = em_height
            # g_beta = 2.0
            # g_left = fcx - alpha / 2.0 * g_beta
            # g_upper = fcy - alpha / 2.0 * g_beta
            # g_right = fcx + alpha / 2.0 * g_beta
            # g_lower = fcy + alpha / 2.0 * g_beta
            # g_face = img_rotated.crop((g_left, g_upper, g_right, g_lower))
            # print(type(g_face))
            # save_path = os.path.join('%s/%s' % (self.params.save_test, name))
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            #
            # g_face = np.float32(g_face)
            # dst = np.zeros(g_face.shape, dtype=np.float32)
            # cv2.normalize(g_face, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
            # # norm_gface = np.uint8(dst * 255)
            # # print(norm_gface.size)
            #
            # resize_img = cv2.resize(dst, (128, 128), cv2.INTER_LINEAR)
            # print('resize后')
            # print(resize_img.shape)
            # g_face = Image.fromarray(np.uint8(resize_img))
            # g_face = g_face.convert('RGB')
            image = np.array(image)
            for face_position in bounding_box:
                face_position = face_position.astype(int)
                x1, y1, x2, y2 = face_position[0], face_position[1], face_position[2], face_position[3]
                gray_face = image[y1:y2, x1:x2]
                print(gray_face.shape)
                plt.imshow(gray_face)
                plt.show()
                try:
                    gray_face = cv2.resize(gray_face, (224, 224))
                except:
                    continue

                gray_face = _preprocess(gray_face)

                # gray_face = Image.fromarray(np.uint8(gray_face))
                # gray_face = gray_face.convert('RGB')
                # gray_face = np.expand_dims(gray_face, 0)
                # gray_face = np.expand_dims(gray_face, -1)


                print(gray_face.shape)
                # g_face = np.expand_dims(g_face, -1)
                gray_face = torch.from_numpy(gray_face).float()
                gray_face_cuda = gray_face.cuda()
                print('gray_face_cuda')
                print(gray_face_cuda)

                out = self.model(gray_face_cuda)
                print(out)

                _, predicted = torch.max(out.data, 1)

                total += label_cuda.long()
                # print(label_cuda.long().data)
                correct += predicted.eq(label_cuda.long()).sum()
                # test_acc = float(correct) / float(total)

                # print(test_acc)

        test_acc = float(correct) / float(total)
        print('total')
        print(total)
        print('correct')
        print(correct)
        self.test_acc.append(test_acc)
        print(self.test_acc)

            # g_face.save(self.params.save_test + '/' + name + '/' + name+'i'+'.jpg')
            # try:
            #    g_face.save(self.params.save_test + '/' + name + '/' + name+'i'+'.jpg')
            #    save_img = True
            # except Exception:
            #     print('save failed')
            #
            # if save_img:
            #     labels.append(label)
            #     imgs.append(g_face)

            # img_labels.append(self.get_image(self.params.save_test + '/' + name + '/' + name+'i'+'.jpg', self.transforms))

            # g_face = Image.fromarray(np.uint8(g_face))
            # g_face = g_face.convert('RGB')
            # plt.imshow(g_face)
            # plt.show()



            # out = self.model(g_face_cuda)

            # g_face = torch.stack(imgs)




                # for idx, img in enumerate(g_face):
                #     image_cuda = img.cuda()
                #     out = self(image_cuda)
                #     _, predicted = torch.max(out.data, 1)
                # for index, i in enumerate(label_dict):
                #     if expression == label_dict[index]:
                #         index = torch.from_numpy(np.asarray(index)).float()
                #         label_cuda = index.cuda()
                #         # label_name = label_name
                #         loss = self.loss(out, label_cuda.long())
                # test_loss += loss.item()
                #
                # total += label_cuda.long().size(0)
                # correct += predicted.eq(label_cuda.long().data).sum()
                # test_acc = float(correct) / float(total)

                # 记录第一次损失
                # if self.test_loss == [] and self.test_acc == []:
                #     self.test_loss.append(test_loss)
                #     if self.params.summary:
                #         self.summary_writer.add_scalar('loss/test_loss', test_loss, 0)
                #         self.summary_writer.add_scalar('acc/test_acc', test_acc, 0)

        # self.pb.close()
        # # test_loss /= total_batc
        # self.test_loss.append(test_loss)
        # # train_acc = correct/len(self.train_loader)
        # print('test correct')
        # print(correct)
        # print('test total')
        # print(total)
        # test_acc = float(correct) / float(total)
        # print('test acc')
        # print(test_acc)
        # self.test_acc.append(test_acc)
        #
        # # add to summary
        # if self.params.summary:
        #     self.summary_writer.add_scalar('loss/test_loss', test_loss, self.epoch)
        #     self.summary_writer.add_scalar('acc/test_acc', test_acc, self.epoch)

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
                print(self.train_loss)
                print(self.val_loss)
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
                print('optimizer')
                print(self.optim.load_state_dict(ckpt['optimizer']))
                print('checkpoint loaded')
                print('Current Epoch: %d' % self.epoch)
                self.checkpoint_flag = True
            except:
                print('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.resume_from)
        else:
            print('Checkpoint do not exists. Start loading pre-trained model......')

    def build_dataloader(self):
        self.dataset = datasets.ImageFolder(self.params.test_data, transform=self.transforms)
        # self.class_to_idx = {k: v for k, v in self.dataset.class_to_idx.items()}
        self.dataset.idx_to_class = {k: v for v, k in self.dataset.class_to_idx.items()}
        self.test_loader = data.DataLoader(self.dataset,
                                           batch_size=self.params.test_batch,
                                           shuffle=False,
                                           collate_fn=lambda x: x[0])

    def get_image(path, trans):
        img = Image.open(path)
        img = trans(img)
        return img


def main():
    params = Params()
    params.test_data = 'datasets/CK+_imgs/test'
    params.save_test = 'datasets/CK+_renew/CK+_re_merge/test_new'
    print('得到h5文件下的数据集')
    # datasets = create_dataset(params)
    testCK = test_CK_net(params)
    # trainCK.Train()
    testCK.Test()

if __name__ == '__main__':
    main()