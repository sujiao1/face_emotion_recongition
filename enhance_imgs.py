from PIL import Image
import os
import random
import cv2
import numpy as np


# imgs_root = 'datasets/fer2013_norm/val'
#
# save_root = 'datasets/fer2013_enhanceimgs/'
# train_enhance_savedir = os.path.join(save_root, 'train')
# validation_enhance_savedir = os.path.join(save_root, 'val')


imgs_root = 'datasets/CK+_renew/CK+_reenhance/train/tail_R1'

save_root = 'datasets/CK+_renew/CK+_re_overfit_enhance/'
train_enhance_savedir = os.path.join(save_root, 'train')
validation_enhance_savedir = os.path.join(save_root, 'val')

def enhance_imgs(imgs_root, save_dir):
    landmarks_tail_dirs = os.listdir(imgs_root)
    print(landmarks_tail_dirs)
    for landmarks_tail_dir in landmarks_tail_dirs:
        if landmarks_tail_dir is not None:
            landmarks_tail_path = os.path.join('%s/%s' % (imgs_root, landmarks_tail_dir))
            print(landmarks_tail_path)
            experssion_dirs = os.listdir(landmarks_tail_path)
            print(experssion_dirs)
            for expression in experssion_dirs:
                if expression == 'surprise':
                    print(expression)
                    imgs_path = os.path.join('%s/%s' % (landmarks_tail_path, expression))
                    print(imgs_path)
                        # m=0
                    for img in os.listdir(imgs_path):
                        img_path = os.path.join('%s/%s' % (imgs_path, img))
                        print(img_path)

                        # img_s = addsalt_pepper(img_path, 0.8)

                        # img_s = PepperandSalt(img_path, 0.8)
                        # gnoises = gasuss_noise(img_path)
                        try:
                            # clahes = claheimg(img_path)
                            # lights_dark = lightness(0.87, img_path)
                            # lights_bright = lightness(1.07, img_path)
                            # rorates = rotate(random.randint(-20,20), img_path)
                            # tranposes = transpose(img_path)
                            # crops = crop(img_path)
                            mblurs = midainblur(img_path)
                            # gnoises = gasuss_noise(img_path)
                            # img_s = addsalt_pepper(img_path, 0.8)
                            # img_s = PepperandSalt(img_path, 0.8)

                        except Exception as e:
                            print(e)

                        # light_dark_img, light_dark_operate = lights_dark
                        # light_bright_img, light_bright_operate = lights_bright
                        # rorate_img, rorate_operate = rorates
                        # tranpose_img, tranpose_operate = tranposes
                        # crop_img, crop_operate = crops
                        # crop_lu_img, crop_ld_img, crop_ru_img, crop_rd_img, \
                        # crop_lu_operate, crop_ld_operate, crop_ru_operate, crop_rd_operate = crops
                        # cll_img, cll_operate = clahes
                        # light_dark_img.show()
                        blur_img, blur_operate = mblurs
                        # gnois_img, gnoise_operate = gnoises
                        # img_salt, salt_operate = img_s

                        # print(img_salt.shape)
                        # print(type(img_salt))
                        # print(type(gnois_img))
                        # print(gnois_img.dtype)
                        # img_salt = img_salt.transpose(2, 1, 0)
                        # cv2.imshow('PepperandSalt', img_salt)
                        # cv2.waitKey(0)
                        # print(img_salt.shape)
                        # print(gnois_img.shape)
                        # print(img_s.cvtColor)
                        # img_salt = cv2.cvtColor(img_salt, cv2.COLOR_BGR2GRAY)
                        # img_salt = Image.fromarray(img_salt)
                        # print(img_salt.shape)
                        # print(img_salt.dtype)
                        # for operate in [light_dark_operate, light_bright_operate, rorate_operate, tranpose_operate, crop_operate]:
                        #     save_path = os.path.join(save_dir, landmarks_tail_dir, operate, expression)
                        #     if not os.path.exists(save_path):
                        #         os.makedirs(save_path)
                        for operate in [blur_operate]:
                            save_path = os.path.join(save_dir, operate, expression)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                        # light_dark_img.save(save_dir + '/' + landmarks_tail_dir + '/' + light_dark_operate + '/' + expression + '/' + img)
                        # light_bright_img.save(save_dir + '/' + landmarks_tail_dir + '/' + light_bright_operate + '/' + expression + '/' + img)
                        # rorate_img.save(save_dir + '/' + landmarks_tail_dir + '/' + rorate_operate + '/' + expression + '/' + img)
                        # tranpose_img.save(save_dir + '/' + landmarks_tail_dir + '/' + tranpose_operate + '/' + expression + '/' + img)
                        # crop_img.save(save_dir + '/' + landmarks_tail_dir + '/' + crop_operate + '/' + expression + '/' + img)
                        # crop_ld_img.save(save_dir + '/' + crop_lu_operate + '/' + expression + '/' + 'crop_lu' + img)
                        # crop_ld_img.save(save_dir + '/' + crop_ld_operate + '/' + expression + '/' + 'crop_ld' + img)
                        # crop_ru_img.save(save_dir + '/' + crop_ru_operate + '/' + expression + '/' + 'crop_ru' + img)
                        # crop_ru_img.save(save_dir + '/' + crop_rd_operate + '/' + expression + '/' + 'crop_rd' + img)
                        # cv2.imwrite(save_dir + '/' + 'clahe' + '/' + expression + '/' + 'clahe' + img, cll_img)
                        # cv2.imwrite(save_dir + '/' + 'gnoise' + '/' + expression + '/' + 'gnoise' + img, gnois_img)
                        cv2.imwrite(save_dir + '/' + blur_operate + '/' + expression + '/' + 'mblur' + img, blur_img)
                        # cv2.imwrite(save_dir + '/' + 'salt_noise' + '/' + expression + '/' + 'salt_noise' + img, img_salt)
                        # img_salt.save(save_dir + '/' + salt_operate + '/' + expression + '/' + 'salt_noise' + img)

def lightness(light, img_path):
    """改变图像亮度.
    推荐值：
        0.87，1.07
    明亮程度
        darker < 1.0 <lighter
    """
    try:
        operate = 'lightness_' + str(light)

        with Image.open(img_path) as image:
            # 图像左右翻转
            light_img = image.point(lambda p: p * light)

        # 日志
        # logger.info(operate)
    except Exception as e:
        print('ERROR %s', operate)
        print(e)
    return light_img, operate


def rotate(angle, img_path):
    """图像旋转15度、30度."""
    try:
        operate = 'rotate'

        with Image.open(img_path) as image:
            # 图像左右翻转
            rorate_img = image.rotate(angle)

        # 日志
        print(operate)
    except Exception as e:
        print('ERROR %s', operate)
        print(e)
    return rorate_img, operate



def transpose(img_path):
    """图像左右翻转操作."""
    try:
        operate = 'transpose'

        with Image.open(img_path) as image:
            # 图像左右翻转
            tranpose_img = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 日志
        print(operate)
    except Exception as e:
        print('ERROR %s', operate)
        print(e)
    return tranpose_img, operate


def deform(img_path):
    """图像拉伸."""
    try:
        operate = 'deform'

        with Image.open(img_path) as image:
            w, h = image.size
            w = int(w)
            h = int(h)
            # 拉伸成宽为w的正方形
            deform_ww = image.resize((int(w), int(w)))

            # 拉伸成宽为h的正方形
            deform_ww = image.resize((int(h), int(h)))

    except Exception as e:
        print('ERROR %s', operate)
        print(e)
    return deform_ww, operate


def crop(img_path):
    """提取四个角落和中心区域."""
    try:
        operate_ld = 'crop_ld'
        operate_ru = 'crop_ru'
        operate_lu = 'crop_lu'
        operate_rd = 'crop_rd'
        # crop_operate = 'crop_c'

        with Image.open(img_path) as image:
            w, h = image.size
            # 切割后尺寸
            scale = 0.875
            # 切割后长宽
            ww = int(w * scale)
            hh = int(h * scale)
            # 图像起点，左上角坐标
            x = y = 0

            # 切割左上角
            x_lu = x
            y_lu = y
            out_lu = image.crop((x_lu, y_lu, ww, hh))
            # savename = self.get_savename(operate + '_lu')
            # out_lu.save(savename, quality=100)
            # print(operate + '_lu')

            # 切割左下角
            x_ld = int(x)
            y_ld = int(y + (h - hh))
            out_ld = image.crop((x_ld, y_ld, ww, hh))
            # savename = self.get_savename(operate + '_ld')
            # out_ld.save(savename, quality=100)
            # print(operate + '_ld')

            # 切割右上角
            x_ru = int(x + (w - ww))
            y_ru = int(y)
            out_ru = image.crop((x_ru, y_ru, w, hh))
            # savename = self.get_savename(operate + '_ru')
            # out_ru.save(savename, quality=100)
            # print(operate + '_ru')

            # # 切割右下角
            x_rd = int(x + (w - ww))
            y_rd = int(y + (h - hh))
            out_rd = image.crop((x_rd, y_rd, w, h))
            # savename = self.get_savename(operate + '_rd')
            # out_rd.save(savename, quality=100)
            # print(operate + '_rd')

            # 切割中心
            x_c = int(x + (w - ww) / 2)
            y_c = int(y + (h - hh) / 2)
            crop_c = image.crop((x_c, y_c, ww, hh))
            # print('提取中心')
    except Exception as e:
        print('ERROR %s', crop_c)
        print(e)
    return out_lu, out_ld, out_ru, out_rd, operate_lu, operate_ld, operate_ru, operate_rd

# 自适应直方图均衡化数据增强
def claheimg(img_path):
    try:
        clahe_operate = 'clahe'

        image = cv2.imread(img_path, 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
        clahe_img = clahe.apply(image)

    except Exception as e:
        print('ERROR %s', clahe_operate)
        print(e)
    return clahe_img, clahe_operate

# 中值滤波
def midainblur(img_path):
    try:
        blur_operate = 'mblur'

        image = cv2.imread(img_path, 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.medianBlur(image, 1)

    except Exception as e:
        print('ERROR %s', blur_operate)
        print(e)
    return blur_img, blur_operate


# 添加高斯噪声
def gasuss_noise(img_path, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    try:
        gnoise_operate = 'gnoise'
        image = cv2.imread(img_path, 0)
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        #cv.imshow("gasuss", out)
    except Exception as e:
           print('Error %s', gnoise_operate)
    return out, gnoise_operate


# 椒盐噪声
def addsalt_pepper(imgpath, SNR):
    """
    添加椒盐噪声
    :param img:
    :param SNR:
    :return:
    """
    try:
        salt_operate = 'salt_pepper'
        img = cv2.imread(imgpath)
        # img_ = img.copy()
        img_ = img.transpose(2, 1, 0)
        c, h, w = img_.shape
        mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
        img_[mask == 1] = 255    # 盐噪声
        img_[mask == 2] = 0      # 椒噪声
    except Exception as e:
        print('Error %s', salt_operate)
    return img_, salt_operate

def PepperandSalt(imgpath, percetage):
    try:
        salt_operate = 'salt_pepper'
        src = cv2.imread(imgpath, 0)
        NoiseImg=src
        NoiseNum=int(percetage*src.shape[0]*src.shape[1])
        for i in range(NoiseNum):
            randX=random.randint(0, src.shape[0]-1)
            randY=random.randint(0, src.shape[1]-1)
            if random.randint(0,1)<=0.5:
                NoiseImg[randX, randY]=0
            else:
                NoiseImg[randX, randY]=255
    except Exception as e:
        print('Error %s', salt_operate)
    return NoiseImg, salt_operate


if __name__ == '__main__':
    # import datetime
    #
    # print('start...')
    # # 计时
    # start_time = datetime.datetime.now()
    #
    # test()
    #
    # end_time = datetime.datetime.now()
    # time_consume = (end_time - start_time).microseconds / 1000000
    enhance_imgs(imgs_root, train_enhance_savedir)
    # img_path = 'datasets/fer2013_landmarks_tail/train/global_tail/0/00000.jpg'
    # img = cv2.imread(img_path, 0)
    # rorate_img = rotate(img, 15)
    # cv2.imshow('rorate_img', rorate_img)
    # cv2.waitKey(0)
    # cv2.imwrite()
