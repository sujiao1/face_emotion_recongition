import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
# from imgaug import augmenters as iaa


# 图像归一化处理
# imgs_root = 'datasets/fer2013_landmarks_tail/val'
#
# save_root = 'datasets/fer2013_norm/'
# train_norm_savedir = os.path.join(save_root, 'train')
# validation_norm_savedir = os.path.join(save_root, 'val')

imgs_root = 'datasets/CK+_renew/CK+_re_overfit_enhance/train'

save_root = 'datasets/CK+_renew/CK+_re_overfit_enhance/outm_norm/'
train_norm_savedir = os.path.join(save_root, 'train')
validation_norm_savedir = os.path.join(save_root, 'val')


def normalization(imgs_root, save_dir):
    landmarks_tail_dirs = os.listdir(imgs_root)
    # print(landmarks_tail_dirs)
    for landmarks_tail_dir in landmarks_tail_dirs:
        if landmarks_tail_dir == 'mblur':
            landmarks_tail_path = os.path.join('%s/%s' % (imgs_root, landmarks_tail_dir))
            # print(landmarks_tail_path)
            experssion_dirs = os.listdir(landmarks_tail_path)
            # print(experssion_dirs)
            for expression in experssion_dirs:
                if expression == 'surprise':
                    # print(expression)
                    imgs_path = os.path.join('%s/%s' % (landmarks_tail_path, expression))
                    # print(imgs_path)
                    # m=0
                    for img in os.listdir(imgs_path):
                        img_path = os.path.join('%s/%s' % (imgs_path, img))
                        print(img_path)
                        try:
                            norm_img = normaliza(img_path)
                            print(norm_img)
                            # print(norm_img.size)
                            # print(norm_img.shape)
                            # if norm_img.dtype != 'uint8':
                            #     print('11111')
                        except Exception as e:
                            print(e)

                        save_path = os.path.join(save_dir, expression)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        cv.imwrite(save_dir + '/' + expression + '/' + img, norm_img)


def normaliza(img_path):
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    # print('cv读取图像')
    # print(img.shape)
    # print(img.size)
    # print(img.dtype)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print('转化为灰度图图像')
    # print(gray_img.shape)
    gray_img = np.float32(gray_img)
    # print('灰度图转换后的dtype')
    # print(gray_img.dtype)

    dst = np.zeros(gray_img.shape, dtype=np.float32)
    cv.normalize(gray_img, dst=dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
    # print('归一化后的图像')
    # print(dst.shape)
    # print(dst.size)
    norm_img = np.uint8(dst * 255)
    print('归一化后转换编码后的dtype')
    print(norm_img.dtype)
    print(norm_img.shape)
    return norm_img



def main():
    normalization(imgs_root, save_root)
    # img_path = 'datasets/fer2013_landmarks_tail/train/global_tail/0/00000.jpg'
    # ori_img = cv.imread(img_path,0)
    # cv.imshow('ori_img', ori_img)
    # cv.waitKey(0)
    #
    # norm_img = normaliza(img_path)
    # cv.imshow('norm_img ', norm_img)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()