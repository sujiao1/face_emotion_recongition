from torchvision import transforms
from torch.utils.data import Dataset
import os
import h5py
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
import torch
import torch.utils.data as data
from sklearn.model_selection import KFold
from src.detector import detect_faces
from mtcnn_landmarks_tailoring import *

# 本脚本主要来创建数据集，读取h5文件获取标签

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

def create_k_dataset(params):
    phase = ['train', 'val', 'test']

    transform = {'train': transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                 'val': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                 'test': transforms.Compose([
                                             transforms.Resize((128, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                 }

    # train_data = h5py.File(os.path.join('%s/%s' % (params.dataset_root, 'CK+_train_K_data.h5')), 'r')
    # train_imgs = np.array(train_data['img'][:])
    # print('train_imgs')
    # print(len(train_imgs))
    # train_labels = np.array(train_data['label'][:])
    # train_dataset = getdata(train_imgs, train_labels, transforms=transform['train'])
    # test_data = h5py.File(os.path.join('%s/%s' % (params.dataset_root, 'CK+_test_data.h5')), 'r')
    # test_imgs = np.array(test_data['img'][:])
    # print('test_imgs')
    # print(len(test_imgs))
    # test_labels = np.array(test_data['label'][:])
    # test_dataset = getdata(test_imgs, test_labels, transforms=transform['test'])
    datasets = {p: getdata(params.dataset_root, kfold=params.kfold, mode=p, transforms=transform[p]) for p in phase}
    # print(datasets['train'])
    # i = 0
    # for imglabel in datasets['train']:
    #     print(imglabel)
    #     i += 1
    # print(i)
    return datasets
    # return train_dataset, test_dataset


class getdata(Dataset):
    def __init__(self, dataset_dir, kfold=2, mode='train', transforms=None):
        # self.imgs = imgs
        # self.labels = labels
        # self.transforms = transforms
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transforms = transforms
        self.kfold = kfold

        # 检查是否有相应的h5数据文件
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError('Unsupported mode %s' % self.mode)

        if not os.path.exists(self.dataset_dir):
            raise ValueError('Dataset not exists at %s' % self.dataset_dir)

        # CK+_train_re_K_data.h5: 21321  CK+_train_outmdg_K_data.h5:17787
        # CK+_train_outm_K_data.h5 17262

        data = h5py.File(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_train_outmdgs128_K_data.h5')), 'r')

        assert kfold > 1
        fold_size = np.array(data['img']).shape[0] // kfold
        # print('fold_size')
        # print(fold_size)
        self.img = np.array(data['img'][:])
        print('数据集大小')
        print(len(self.img))
        self.label = np.array(data['label'][:])
        folds = list(KFold(n_splits=kfold, shuffle=True, random_state=1).split(self.img, self.label))
        # print('folds')
        # print(folds)
        for j, (train_idx, val_idx) in enumerate(folds):
            self.train_img = self.img[train_idx]
            self.train_label = self.label[train_idx]
            self.val_img = self.img[val_idx]
            self.val_label = self.label[val_idx]
            # print('train_img_len')
            # print(self.train_img)
            # print(len(self.train_img))
            # print('train_label_len')
            # print(self.train_label)
            # print(len(self.train_label))
            # print('val_img_len')
            # print(len(self.val_img))
            # print(self.val_img)
            # print('val_label')
            # print(self.val_label)

        if self.mode == 'train':
            self.img = self.train_img
            self.label = self.train_label
        elif self.mode == 'val':
            self.img = self.val_img
            self.label = self.val_label
        else:
            data = h5py.File(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_test128_K_data.h5')),
                             'r')
            self.img = np.array(data['img'][:])
            self.label = np.array(data['label'][:])

        # img = torch.from_numpy(self.img).float()
        # # print(len(img))
        # print('img')
        # print(img)
        # label = torch.from_numpy(self.label).float()
        # print('label')
        # print(label)
        # self.train_img, self.train_label = None, None
        # for i in range(kfold):
        #     for j in range(kfold):
        #         idx = slice(j * fold_size, (j+1) * fold_size)
        #         print('idx')
        #         print(idx)
        #         img_part, label_part = img[idx, :], label[idx]
        #         print('img_part')
        #         print(img_part)
        #         print('label_part')
        #         print(label_part)
        #         if j == i:
        #             self.val_img, self.val_label = img_part, label_part
        #             print('len_img')
        #             print(len(self.val_img))
        #             print('val_img')
        #             print(self.val_img)
        #             print('val_label')
        #             print(self.val_label)
        #         elif self.train_img is None:
        #             self.train_img, self.train_label = img_part, label_part
        #         else:
        #             self.train_img = torch.cat((self.train_img, img_part), dim=0)
        #             self.train_label = torch.cat((self.train_label, label_part), dim=0)

    def __getitem__(self, index):
        if self.mode == 'train':
            # img, target = self.train_data[index], self.train_label[index]
            img = torch.from_numpy(self.img[index]).float()
            # print('img')
            # print(img)
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            # print('label')
            # print(label)
            label_name = label_dict[self.label[index]]
            # print('label_name')
            # print(label_name)
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            # print('np.concatenate')
            # print(img.shape)
            img = Image.fromarray(np.uint8(img))
            img = img.convert('RGB')
            # print(img.shape)
            plt.imshow(img)
            # plt.show()
            if self.transforms is not None:
                img = self.transforms(img)
                # print('tranform之后')
                # print(img.shape)
                # print(img)
            return img, label, label_name
        elif self.mode == 'val':
            img = torch.from_numpy(self.img[index]).float()
            # print('img')
            # print(img)
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            # print('label')
            # print(label)
            label_name = label_dict[self.label[index]]
            # print('label_name')
            # print(label_name)
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            # print('np.concatenate')
            # print(img.shape)
            img = Image.fromarray(np.uint8(img))
            img = img.convert('RGB')
            # print(img.shape)
            plt.imshow(img)
            # plt.show()
            if self.transforms is not None:
                img = self.transforms(img)
                # print('tranform之后')
                # print(img.shape)
                # print(img)
            return img, label, label_name
        else:
            img = torch.from_numpy(self.img[index]).float()
            # print('img')
            # print(img)
            # print('每一个label')
            # print(self.label[index])
            # label_array = np.asarray(self.label[index])
            # print('转换为数组之后的label')
            # print(label_array)
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            print('转换为tensor')
            print(label)
            label_name = label_dict[self.label[index]]
            # print('label_name')
            # print(label_name)
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            # print('np.concatenate')
            # print(img.shape)
            img = Image.fromarray(np.uint8(img))
            img = img.convert('RGB')
            if len(img.size) == 2:
                img = img.convert('RGB')
            else:
                img = img

            h, w = img.size
            bounding_box, landmarks = detect_faces(img)
            print('landmarks')
            print(landmarks)
            print(bounding_box.shape)
            faces_count = landmarks.shape[0]
            if faces_count > 1:
                print('图片中有多个人脸')
                return
            elx, erx, nx, mlx, mrx, ely, ery, ny, mly, mry = landmarks[0]

            # 计算旋转角度
            angle = calculate_angle(elx, ely, erx, ery)

            # 旋转图像
            img_rotated = img.rotate(angle, expand=1)
            ww, hh = img_rotated.size  # 旋转后图像的宽度和高度

            # 对齐后的位置
            elx, ely = pos_transform(angle, elx, ely, w, h)
            erx, ery = pos_transform(angle, erx, ery, w, h)
            nx, ny = pos_transform(angle, nx, ny, w, h)
            mlx, mly = pos_transform(angle, mlx, mly, w, h)
            mrx, mry = pos_transform(angle, mrx, mry, w, h)

            # 基本参数
            eye_width = erx - elx  # 两眼之间的距离
            ecx, ecy = (elx + erx) / 2.0, (ely + ery) / 2.0  # 两眼中心坐标
            mouth_width = mrx - mlx  # 嘴巴的宽度
            mcx, mcy = (mlx + mrx) / 2.0, (mly + mry) / 2.0  # 嘴巴中心坐标
            em_height = mcy - ecy  # 两眼睛中心到嘴巴中心高度
            fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # 人脸中心坐标

            # 纯脸
            if eye_width > em_height:
                alpha = eye_width
            else:
                alpha = em_height
            g_beta = 2.0
            g_left = fcx - alpha / 2.0 * g_beta
            g_upper = fcy - alpha / 2.0 * g_beta
            g_right = fcx + alpha / 2.0 * g_beta
            g_lower = fcy + alpha / 2.0 * g_beta
            g_face = img_rotated.crop((g_left, g_upper, g_right, g_lower))

            # print('g_face')
            # print(type(g_face))
            # print(img.size)

            # g_face = Image.fromarray(np.uint8(g_face))
            # g_face = g_face.convert('RGB')
            plt.imshow(g_face)
            plt.show()
            if self.transforms is not None:
                g_face = self.transforms(g_face)
                # print('tranform之后')
                # print(img.shape)
                # print(img)
            return g_face, label, label_name

    def __len__(self):
        if self.mode == 'train':
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]
        elif self.mode == 'val':
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]
        else:
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]


def main():
    params = Params()
    params.dataset_root = 'datasets/img_label_data'
    params.kfold = 10
    # create_dataset(params)
    datasets = create_k_dataset(params)
    # train_loader = data.DataLoader(dataset=datasets['train'], num_workers=4, batch_size=16, shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(dataset=datasets['test'], batch_size=16, shuffle=False, pin_memory=True)
    # for (img, label, label_name) in train_loader:
    #     print('train_loader')
    #     print(img.size())
    #     print(label.size())
    # print(train_loader)
    # print(len(train_loader))
    for step, (img, label, label_name) in enumerate(test_loader):
        print('step train_loader')
        print(img.shape)
        print(label.shape)


if __name__ == '__main__':
    main()






