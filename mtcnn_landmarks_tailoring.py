# -*- coding: utf- 8 -*-
from PIL import Image, ImageDraw
from src.detector import detect_faces
import math
import os
import cv2

# imgs_root = 'datasets/fer2013imgs/val'
# training_csv_file = 'datasets/fer2013_file/train.csv'
# validation_csv_file = 'datasets/fer2013_file/val.csv'
#
# save_root = 'datasets/fer2013_landmarks_tail/'
# train_savedir = os.path.join(save_root, 'train')
# validation_savedir = os.path.join(save_root, 'val')

imgs_root = 'datasets/fer2013imgs/train'

save_root = 'datasets/CK+_renew/CK+_re_merge/test/'
train_savedir = os.path.join(save_root, 'train')
validation_savedir = os.path.join(save_root, 'val')

def mtcnn_landmarks_tail(images_root, save_dir):
    expression_dirlist = os.listdir(images_root)
    print(expression_dirlist)
    for expression in expression_dirlist:
        images_path = os.path.join('%s/%s' % (images_root, expression))
        print(images_path)
        # m=0
        # images = os.listdir(images_path)
        # print(images)
        for img in os.listdir(images_path):
            # print(expression)
            # print(img)
            # if expression == 0:
            img_path = os.path.join('%s/%s/%s' % (images_root, expression, img))
            print(img_path)
            # print(img.__sizeof__())
            # img_tails = crop_face(img_path)
            try:
                img_tails = crop_face(img_path)
            except Exception:
                print('** Get Some Errors, Skip {} **'.format(img_path))
                continue

            # faces = crop_face(img_path)
            if not img_tails:
                continue

            image_tail, global_tail, tail_R1, tail_R2 = img_tails

            for part in ['image_tail', 'global_tail', 'tail_R1', 'tail_R2']:
                save_path = os.path.join(save_dir, part, expression)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            # print(save_dir + "/" + 'image_tail' + "/" + expression + "/" + img)
            # image_tail.save(save_dir + "/" + 'image_tail' + "/" + expression + "/" + img)
            # global_tail.save(save_dir + "/" + 'global_tail' + "/" + expression + "/" + img)
            # tail_R1.save(save_dir + "/" + 'tail_R1' + "/" + expression + "/" + img)
            # tail_R2.save(save_dir + "/" + 'tail_R2' + "/" + expression + "/" + img)


# cv读取图像可以用img.shape来输出维度(h,w,c)，此处为三通道图像,img.size输出一个值, type类型：<class 'numpy.ndarray'>,dtype编码为uint8
# PIL用Image读取img.size输出(w,h),因为PIL读取非array类型，为object类型，所以无法用img.shape进行输出, type类型:<class 'PIL.Image.Image'>,无法读取dtype编码类型
def crop_face(imgpath):
    # img = cv2.imread(imgpath)
    # print(img.size)
    # print(img.shape)
    # print(type(img))
    img = Image.open(imgpath)
    if len(img.size) == 2:
        image = img.convert('RGB')
    else:
        image = img
    h, w = image.size  # 图像的宽度和高度
    print(image.size)
    print(type(image))
    # print(img.dtype)
    bounding_box, landmarks = detect_faces(image)
    print('landmarks')
    print(landmarks)
    print(bounding_box.shape)
    faces_count = landmarks.shape[0]
    if faces_count > 1:
        print('图片中有多个人脸')
        return
    # 左眼中心位置: (elx, ely)
    # 右眼中心位置: (erx, ery)
    # 鼻尖位置: (nx, ny)
    # 左嘴角位置: (mlx, mrx)
    # 右嘴角位置: (mrx. mry)
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

    # draw = ImageDraw.Draw(img_rotated)

    # # 在图像上画出眼睛，鼻子，嘴角的位置
    # r = 3
    # draw.ellipse([(elx - r, ely - r), (elx + r, ely + r)], fill='red')
    # draw.ellipse([(erx - r, ery - r), (erx + r, ery + r)], fill='red')
    # draw.ellipse([(nx - r, ny - r), (nx + r, ny + r)], fill='red')
    # draw.ellipse([(mlx - r, mly - r), (mlx + r, mly + r)], fill='red')
    # draw.ellipse([(mrx - r, mry - r), (mrx + r, mry + r)], fill='red')

    # draw.line([(elx, ely), (erx, ery)], fill='blue')
    # draw.line([(elx, ely), (mrx, mry)], fill='blue')
    # draw.line([(erx, ery), (mlx, mly)], fill='blue')
    # draw.line([(mlx, mly), (mrx, mry)], fill='blue')
    # # img_rotated.show()
    # print(img_rotated.shape)

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

    # 将人脸从鼻子位置处划分成上下两部分，分别为R1区域和R2区域
    # 空间归一化采用眼睛之间的距离(α)的一半
    alpha = (erx - elx) / 2.0  # 空间归一化距离
    crop_width = alpha * 3.8  # 水平因子2.4
    crop_R1_height = alpha * 1.5  # R1区域高度，垂直因子1.5
    crop_R2_height = alpha * 3.5  # R2区域高度，垂直因子3.5

    ecx, ecy = (elx + erx) / 2.0, ely # 两眼中心点位置
    # 计算裁剪位置，左上角，右上角，左下角，右下角点位置
    TL_x, TL_y = ecx - crop_width / 2.0, ecy - crop_R1_height
    TR_x, TR_y = ecx + crop_width / 2.0, ecy - crop_R1_height
    BL_x, BL_y = ecx - crop_width / 2.0, ecy + crop_R2_height
    BR_x, BR_y = ecx + crop_width / 2.0, ecy + crop_R2_height

    left, upper, right, lower = get_crop_area(TL_x, TL_y, TR_x, BL_y, ww, hh)
    img_crop = img_rotated.crop((left, upper, right, lower))
    # img_crop.show()

    # # 画出裁剪位置
    # draw.line([(TL_x, TL_y), (TR_x, TR_y)], fill='green')
    # draw.line([(TL_x, TL_y), (BL_x, BL_y)], fill='green')
    # draw.line([(TR_x, TR_y), (BR_x, BR_y)], fill='green')
    # draw.line([(BL_x, BL_y), (BR_x, BR_y)], fill='green')

    # R1裁剪区域，左下角和右下角位置坐标
    alpha_h = 0.8
    alpha_v = 1.0
    R1_TL_x, R1_TL_y = elx - alpha * alpha_h, ely - alpha * alpha_v
    R1_TR_x, R1_TR_y = erx + alpha * alpha_h, ely - alpha * alpha_v
    R1_BL_x, R1_BL_y = elx - alpha * alpha_h, ely + alpha * alpha_v
    R1_BR_x, R1_BR_y = erx + alpha * alpha_h, ery + alpha * alpha_v
    if R1_TL_y > R1_TR_y:
        R1_TL_y = R1_TR_y
    if R1_BL_y < R1_BR_y:
        R1_BL_y = R1_BR_y
    left, upper, right, lower = get_crop_area(R1_TL_x, R1_TL_y, R1_TR_x, R1_BL_y, ww, hh)
    R1 = img_rotated.crop((left, upper, right, lower))
    # R1.show()

    # # 画出R1裁剪区域
    # draw.line([(R1_TL_x, R1_TL_y), (R1_TR_x, R1_TR_y)], fill='blue')
    # draw.line([(R1_TL_x, R1_TL_y), (R1_BL_x, R1_BL_y)], fill='blue')
    # draw.line([(R1_BL_x, R1_BL_y), (R1_BR_x, R1_BR_y)], fill='blue')
    # draw.line([(R1_TR_x, R1_TR_y), (R1_BR_x, R1_BR_y)], fill='blue')

    # R2裁剪区域 左上角和右上角位置坐标
    # 空间归一化采用嘴角距离
    beta = mrx - mlx
    beta_h = 0.3
    beta_v = 0.4
    R2_TL_x, R2_TL_y = mlx - beta * beta_h, mly - beta * beta_v
    R2_TR_x, R2_TR_y = mrx + beta * beta_h, mry - beta * beta_v
    R2_BL_x, R2_BL_y = mlx - beta * beta_h, mly + beta * beta_v
    R2_BR_x, R2_BR_y = mrx + beta * beta_h, mry + beta * beta_v

    if R2_TL_y > R2_TR_y:
        R2_TL_y = R2_TR_y
    if R2_BL_y < R2_BR_y:
        R2_BL_y = R2_BR_y
    left, upper, right, lower = get_crop_area(R2_TL_x, R2_TL_y, R2_TR_x, R2_BL_y, ww, hh)
    R2 = img_rotated.crop((left, upper, right, lower))
    # R2.show()

    # draw.line([(R2_TL_x, R2_TL_y), (R2_TR_x, R2_TR_y)], fill='blue')
    # draw.line([(R2_TL_x, R2_TL_y), (R2_BL_x, R2_BL_y)], fill='blue')
    # draw.line([(R2_TR_x, R2_TR_y), (R2_BR_x, R2_BR_y)], fill='blue')
    # draw.line([(R2_BL_x, R2_BL_y), (R2_BR_x, R2_BR_y)], fill='blue')
    # img_rotated.show()
    return img_rotated, g_face, R1, R2


def calculate_angle(elx, ely, erx, ery):
    dx = erx - elx
    dy = ery - ely
    angle = math.atan(dy / dx) * 180 / math.pi
    return angle


def pos_transform(angle, ex, ey, w, h):  # (x, y)为要旋转的点， w, h 图像的宽度和高度
    angle = angle * math.pi / 180
    matrix = [
        math.cos(angle), math.sin(angle), 0.0,
        -math.sin(angle), math.cos(angle), 0.0
    ]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    # 计算输出图像大小
    xx = []
    yy = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        x, y = transform(x, y)
        xx.append(x)
        yy.append(y)
    ww = int(math.ceil(max(xx)) - math.floor(min(xx)))
    hh = int(math.ceil(max(yy)) - math.floor(min(yy)))

    # 调整图像中心位置
    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = ww / 2.0 - cx
    matrix[5] = hh / 2.0 - cy
    return transform(ex, ey)


def get_crop_area(left, upper, right, lower, w, h):
    """
    保证裁剪区域不超出图像区域
    :param left:
    :param upper:
    :param right:
    :param lower:
    :param w: 图像宽度
    :param h: 图像高度
    :return:
    """
    if left < 0:
        left = 0
    if upper < 0:
        upper = 0
    if right > w:
        right == w
    if lower > h:
        lower = h
    return left, upper, right, lower


if __name__ == '__main__':
    # crop_face('datasets/fer2013_landmarks/train/image_rote/0/00000.jpg')
    mtcnn_landmarks_tail(imgs_root, train_savedir)