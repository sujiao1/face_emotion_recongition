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

def create_dataset(params):
    phase = ['train', 'val', 'test']

    transform = {'train': transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                 'val': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                 'test': transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                 }

    # dataset = {p: Getdata(params.dataset_root, mode=p, transforms=transform[p]) for p in phase}
    datasets = {p: getdata(params.dataset_root, mode=p, transforms=transform[p]) for p in phase}
    print(datasets['train'])
    # i = 0
    # for imglabel in datasets['train']:
    #     print(imglabel)
    #     i += 1
    # print(i)
    return datasets


class getdata(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms=None):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.mode = mode

        # 检查是否有相应的h5数据文件
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError('Unsupported mode %s' % self.mode)

        if not os.path.exists(self.dataset_dir):
            raise ValueError('Dataset not exists at %s' % self.dataset_dir)

        # 根据要求的文件不同，读取不同的h5文件
        if self.mode == 'train':
            # self.fo = open(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_train_data.h5')), 'r')
            # self.data = pickle.dumps(self.fo)
            # self.data = pickle.loads(self.data)
            data = h5py.File(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_train_data.h5')),
                                  'r')
            self.img = np.array(data['img'][:])
            # print('img')
            # print(self.img)
            self.label = np.array(data['label'][:])
            # print('label')
            # print(self.label)

            # self.data = pickle.dumps(self.data, protocol=1)
            # self.data = pickle.loads(self.data)
            # train_index = []
            # number = len(self.data['label'])
            #
            # print(number)
            # for i in range(number):
            #     train_index.append(i)
            # print(train_index)
            #
            # self.train_data = []
            # self.train_label = []
            # for index in range(len(train_index)):
            #     self.train_data.append(self.data['img'][train_index[index]])
            #     self.train_label.append(self.data['label'][train_index[index]])
            #     img, target = self.train_data[index], self.train_label[index]
        elif self.mode == 'val':
            data = h5py.File(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_val_data.h5')),
                                  'r')
            self.img = np.array(data['img'][:])
            self.label = np.array(data['label'][:])
            # val_index = []
            # number = len(self.data['label'])
            # print(number)
            # for i in range(number):
            #     val_index.append(i)
            # print(val_index)
            #
            # self.val_data = []
            # self.val_label = []
            # for index in range(len(val_index)):
            #     self.val_data.append(self.data['img'][val_index[index]])
            #     self.val_label.append(self.data['label'][val_index[index]])
            #     img, target = self.val_data[index], self.val_label[index]
        else:
            data = h5py.File(os.path.join('%s/%s' % (self.dataset_dir, 'CK+_test_data.h5')),
                             'r')
            self.img = np.array(data['img'][:])
            self.label = np.array(data['label'][:])

    def __getitem__(self, index):
        if self.mode == 'train':
            # img, target = self.train_data[index], self.train_label[index]
            img = torch.from_numpy(self.img[index]).float()
            print('img')
            print(img)
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            print('label')
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
            # print(img.shape)
            plt.imshow(img)
            # plt.show()
            if self.transforms is not None:
                img = self.transforms(img)
                print('tranform之后')
                print(img.shape)
                print(img)
            return img, label, label_name
        elif self.mode == 'val':
            img = torch.from_numpy(self.img[index]).float()
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            label_name = label_dict[self.label[index]]
            # img, target = self.val_data[index], self.val_label[index]
            # 将图片转换为PIL img
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            # print('np.concatenate')
            # print(img.shape)
            img = Image.fromarray(np.uint8(img))
            img = img.convert('RGB')
            # # print(img.shape)
            # # plt.imshow(img)
            # # plt.show()
            plt.imshow(img)
            # plt.show()
            if self.transforms is not None:
                img = self.transforms(img)
                # print('tranform之后')
                # print(img.shape)
            return img, label, label_name
        else:
            # img, target = self.test_data[index],self.test_label[index]
            img = torch.from_numpy(self.img[index]).float()
            label = torch.from_numpy(np.asarray(self.label[index])).float()
            label_name = label_dict[self.label[index]]
            # img, target = self.val_data[index], self.val_label[index]
            # 将图片转换为PIL img
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            # print('np.concatenate')
            # print(img.shape)
            img = Image.fromarray(np.uint8(img))
            img = img.convert('RGB')
            # # print(img.shape)
            # # plt.imshow(img)
            # # plt.show()
            plt.imshow(img)
            # plt.show()
            if self.transforms is not None:
                img = self.transforms(img)
                # print('tranform之后')
                # print(img.shape)
            return img, label, label_name

    def __len__(self):
        if self.mode == 'train':
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]
        elif self.mode == 'val':
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]
        else:
            # return len(self.test_data)
            assert self.label.shape[0] == self.img.shape[0]
            return self.label.shape[0]


def main():
    params = Params()
    params.dataset_root = 'datasets/img_label_data'
    create_dataset(params)
    datasets = create_dataset(params)
    trainset = datasets['train']
    train_loader = data.DataLoader(dataset=trainset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    for (img, label) in train_loader:
        print('train_loader')
        print(img.size())
        print(label.size())
    print(train_loader)
    print(len(train_loader))
    for step, (img, label) in enumerate(train_loader):
        print('step train_loader')
        print(img.shape)
        print(label.shape)


if __name__ == '__main__':
    main()






