import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

imgs_root = 'datasets/fer2013_enhanceimgs/val/tail_R2'

save_root = 'datasets/fer2013_preimgs/val/tail_R2/'
# train_pre_savedir = os.path.join(save_root, 'train')
# validation_pre_savedir = os.path.join(save_root, 'val')

# imgs_root = 'datasets/CK+_norm/val'
#
# save_root = 'datasets/CK+_norm_preimgs/'
# train_pre_savedir = os.path.join(save_root, 'train')
# validation_pre_savedir = os.path.join(save_root, 'val')

def pretreatment(imgs_root, save_dir):
    landmarks_tail_dirs = os.listdir(imgs_root)
    print(landmarks_tail_dirs)
    for landmarks_tail_dir in landmarks_tail_dirs:
        landmarks_tail_path = os.path.join('%s/%s' % (imgs_root, landmarks_tail_dir))
        print(landmarks_tail_path)
        experssion_dirs = os.listdir(landmarks_tail_path)
        print(experssion_dirs)
        for expression in experssion_dirs:
            print(expression)
            imgs_path = os.path.join('%s/%s' % (landmarks_tail_path, expression))
            # m=0
            for img in os.listdir(imgs_path):
                img_path = os.path.join('%s/%s' %(imgs_path, img))
                print(img_path)
                image = cv.imread(img_path, 0)  # 直接读为灰度图像
                print(image.shape)

                clahe_img = pretreat_CLAHE(image)
                canny_img = pre_canny(image)
                medianblur_img = pretreat_medianblur(image)
                clahe_medianblur_img = pretreat_medianblur(clahe_img)
                sobel_img = pre_sobel(image)
                robert_img = pre_roberts(image)
                prewitt_img = pre_prewitt(image)
                laplacian_img = pretreat_laplacian(image)
                medianblur_sobel_img = pre_sobel(medianblur_img)
                medianblur_laplacian_img = pretreat_laplacian(medianblur_img)


                for pretreat in ['clahe', 'canny', 'medianblur', 'clahe_medianblur', 'sobel', 'robert', 'prewitt', 'laplacian','medianblur_sobel', 'medianblur_laplacian']:
                    save_path = os.path.join(save_dir, landmarks_tail_dir, pretreat, expression)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)


                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'clahe' + '/' + expression + '/' + img, clahe_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'canny' + '/' + expression + '/' + img, canny_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'medianblur' + '/' + expression + '/' + img, medianblur_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'clahe_medianblur' + '/' + expression + '/' + img, clahe_medianblur_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'sobel' + '/' + expression + '/' + img, sobel_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'robert' + '/' + expression + '/' + img, robert_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'prewitt' + '/' + expression + '/' + img, prewitt_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'laplacian' + '/' + expression + '/' + img, laplacian_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'medianblur_sobel' + '/' + expression + '/' + img, medianblur_sobel_img)
                cv.imwrite(save_dir + '/' + landmarks_tail_dir + '/' + 'medianblur_laplacian' + '/' + expression + '/' + img, medianblur_laplacian_img)

# 自适应直方图均衡化
def pretreat_CLAHE(img):
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    clahe_img = clahe.apply(img)
    return clahe_img

# 中值滤波
def pretreat_medianblur(img):
    blur_img = cv.medianBlur(img, 1)
    return blur_img

# 拉普拉斯滤波
def pretreat_laplacian(img):
    # laplacian_img = cv.Laplacian(img, cv.CV_64F)
    dst = cv.Laplacian(img, cv.CV_16S, ksize=1)
    laplacian_img = cv.convertScaleAbs(dst)
    return laplacian_img


# 边缘检测  https://blog.csdn.net/Eastmount/article/details/89001702
# sobel算子提取边缘
def pre_sobel(img):

    x = cv.Sobel(img, cv.CV_16S, 1, 0)
    y = cv.Sobel(img, cv.CV_16S, 0, 1)

    # 转回uint8
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)

    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

# https://blog.csdn.net/HuangZhang_123/article/details/80511270?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
# 边缘检测及轮廓检测
def pre_canny(img):
    canny_img = cv.Canny(img, 200, 300)
    return canny_img

# 锐化
def pre_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    sharpen_img = cv.filter2D(img, -1, kernel=kernel)
    return sharpen_img

# robet边缘检测
def pre_roberts(img):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv.filter2D(img, cv.CV_16S, kernelx)
    y = cv.filter2D(img, cv.CV_16S, kernely)
    # 转uint8
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Roberts_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts_img

# prewitt边缘检测
def pre_prewitt(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv.filter2D(img, cv.CV_16S, kernelx)
    y = cv.filter2D(img, cv.CV_16S, kernely)
    # 转uint8
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    prewitt_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    return prewitt_img


def main():
    pretreatment(imgs_root, save_root)
    # 直接读为灰度图像  numpy.array
    """
    img = cv.imread('datasets/CK+_enhanceimgs/train/global_tail/crop/anger/S010_004_00000019.png', 0)
    print(img.size)
    print('原图')
    cv.imshow('orignal', img)
    cv.waitKey(0)

    print('自适应直方图均衡1')
    clahe_img = pretreat_CLAHE(img)
    print(clahe_img.size)
    cv.imshow('clahe_img', clahe_img)
    cv.waitKey(0)

    print('中值滤波1')
    medianblur_img = pretreat_medianblur(img)
    print(medianblur_img.size)
    cv.imshow('medianblur_img', medianblur_img)
    cv.waitKey(0)

    print('直方图均衡化后进行中值滤波1')
    medianblurt_img = pretreat_medianblur(clahe_img)
    print(medianblurt_img.size)
    cv.imshow('medianblurt_img', medianblurt_img)
    cv.waitKey(0)

    print('直方图均衡化后进行中值滤波后进行sobel算子边缘检测0')
    clahe_median_sobel_img = pre_sobel(medianblurt_img)
    print(clahe_median_sobel_img.size)
    cv.imshow('clahe_median_sobel_img', clahe_median_sobel_img)
    cv.waitKey(0)

    print('拉普拉斯滤波1')
    laplacian_img = pretreat_laplacian(img)
    print(laplacian_img.size)
    cv.imshow('laplacian_img', laplacian_img)
    cv.waitKey(0)

    print('中值滤波后拉普拉斯滤波1')
    meidanblur_laplacian_img = pretreat_laplacian(medianblur_img)
    print(meidanblur_laplacian_img.size)
    cv.imshow('meidanblur_laplacian_img', meidanblur_laplacian_img)
    cv.waitKey(0)

    print('sobel算子进行边缘检测1')
    sobel_img = pre_sobel(img)
    print(sobel_img.size)
    cv.imshow('sobel_img', sobel_img)
    cv.waitKey(0)

    print('中值滤波之后进行sobel边缘检测1')
    sobelt_img = pre_sobel(medianblur_img)
    print(sobelt_img.size)
    cv.imshow('sobelt_img', sobelt_img)
    cv.waitKey(0)

    print('sobel算子边缘检测之后进行中值滤波1')
    medianblurr_img = pretreat_medianblur(sobel_img)
    print(medianblurr_img.size)
    cv.imshow('medianblurr_img', medianblurr_img)
    cv.waitKey(0)

    print('canny边缘检测1')
    canny_img = cv.Canny(img, 200, 300)
    print(canny_img.size)
    cv.imshow('canny_img', canny_img)
    cv.waitKey(0)

    print('Robert边缘检测1')
    roberts_img = pre_roberts(img)
    print(roberts_img.size)
    cv.imshow('roberts_img', roberts_img)
    cv.waitKey(0)

    print('prewitt边缘检测1')
    prewitt_img = pre_prewitt(img)
    print(prewitt_img.size)
    cv.imshow('prewitt_img', prewitt_img)
    cv.waitKey(0)


    print('fliter2D锐化0')
    sharpen_img = cv.Canny(img, 200, 300)
    print(sharpen_img.size)
    cv.imshow('sharpen_img', sharpen_img)
    cv.waitKey(0)
    """

if __name__ == '__main__':
    main()