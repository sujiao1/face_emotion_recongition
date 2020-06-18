import numpy as np
import torch
from torch.autograd import Variable
from src.get_nets import PNet, RNet, ONet
from src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from src.first_stage import run_first_stage


def detect_faces(image, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.
        nms_threshold: 三次非极大值抑制筛选人脸框的IOU阈值，
        三个网络可以分别设置，值设置的过小，nms合并的太少，会产生较多的冗余计算。
        threshold：人脸框得分阈值，三个网络可单独设定阈值，
        值设置的太小，会有很多框通过，也就增加了计算量，
        还有可能导致最后不是人脸的框错认为人脸。
        min_face_size: 最小可检测图像，该值大小，可控制图像金字塔的阶层数的参数之一，
        越小，阶层越多，计算越多。



    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    """
    得到的results是一个长度为2的tuple类型数据，其中results[0]是N*5的numpy array，
    表示人脸的bbox信息，其中N表示检测到的人脸数量，5表示每张人脸有4个坐标点（左上角的x，y和右下角的x，y）和1个置信度score。
    results[1]是N*10的numpy array，表示人脸关键点信息，其中N表示检测到的人脸数量，10表示5个关键点的x、y坐标信息。
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    # 生成图像金字塔

    # 缩放到12为止
    # 代表PNet的输入图像长宽，都为12
    min_detection_size = 12
    # factor：生成图像金字塔时候的缩放系数, 范围(0,1)，
    # 可控制图像金字塔的阶层数的参数之一，越大，阶层越多，计算越多。本文取了0.707。
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    # min_face_size:最小可检测图像：20
    m = min_detection_size/min_face_size
    # image图片的初始缩放尺寸，非规定的尺寸，针对图片的真是尺寸缩放，按照规定缩放尺寸/最小检测图像计算
    min_length *= m

    # 金字塔层数
    # scales这个vector保存的是每次缩放的系数，它的尺寸代表了可以缩放出的图片的数量。
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    #
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    # 长度为len(batch)的list，list中的每个numpy array表示对应scale的bbox信息，
    # 每个numpy array的shape为K*9，K就是bbox的数量，9包含4个坐标点信息，
    # 一个置信度score和4个用来调整前面4个坐标点的偏移信息。最后都并到bounding_boxes列表中，
    # 因此该列表一共包含len(scales)个尺度的numpy array，
    # 但是由于该列表中某些值是None，所以会有去掉None的操作。
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    # 将由numpy array组成的list按照列叠加成一个新的numpy array格式的bounding_boxes，
    # 这个新的bounding_boxes依然是2维的，每一行代表一个bbox，一共9列。
    # 去掉空
    bounding_boxes = np.vstack(bounding_boxes)

    # nms该函数返回的pick是一个list，list中的值是index，这些index是非重复的index
    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    # 将bounding_boxes中的这些非重复的框挑选出来。
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    # https://blog.csdn.net/wfei101/article/details/79918237
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    # 将bounding_boxes的尺寸调整为正方形
    bounding_boxes = convert_to_square(bounding_boxes)
    # 对四个坐标点的取整操作。也就是说bounding_boxes是N*5的numpy array，N表示bbox的数量。
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    # 自动求导机制
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    # 输出output是一个长度为2的list，其中output[0]是大小为N*4的numpy array，表示N个bbox的回归信息；
    # output[1]是大小为N*2的numpy array，表示N个bbox的类别信息。
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    probs = output[1].data.numpy()  # shape [n_boxes, 2]

    # 通过比较某个bbox属于人脸的概率和阈值来判断该bbox是否是人脸。通过这一步就可以过滤掉大部分的非人脸bbox。
    # keep:人脸索引
    # 将人脸概率信息也添加到bounding_boxes中，相当于score。
    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    # 根据回归信息reg来调整bounding_boxes中bbox的坐标信息，
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    # 将4个坐标值从float64转成整数。
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    # 生成的output是一个长度为3的list
    # 其中output[0]是N*10的numpy array，表示每个bbox的5个关键点的x、y坐标相关信息，
    # 剩下的output[1]和output[2]和second stage类似，分别表示回归信息和分类信息
    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    # 计算landmark point部分，因为前面得到的关键点的x、y坐标相关信息并不直接是x、y的值，而是一个scale值，
    # 最终的关键点的x、y值可以通过这个scale值和bbox的宽高相乘再累加到bbox的坐标得到，具体而言就是下面这两行代码
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]
    # results（return）是一个长度为2的tuple，
    # 其中result[0]是人脸框的坐标和置信度信息，是一个N*5的numpy array；
    # result[1]是人脸关键点信息，是一个N*10的numpy array。
    return bounding_boxes, landmarks
