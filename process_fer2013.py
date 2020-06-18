# create data and label for CK+
#  0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=contempt
# contain 135,177,75,207,84,249,54 images

import os
import numpy as np
import h5py
import skimage.io
import cv2

CK_train_path = 'datasets/CK+_K_data/train_K_data'

# def unpickle(file):
#     import pickle
#     fo = open(file, 'rb')
#     dict = pickle.load(fo)
#     return dict

def emotion_label(emotion_path):
    anger_path = os.path.join('%s/%s' % (emotion_path, 'anger'))         # train-450        1350   val-162  297
    contempt_path = os.path.join('%s/%s' % (emotion_path, 'contempt'))   # train-180  480   1440   val-72   192
    disgust_path = os.path.join('%s/%s' % (emotion_path, 'disgust'))     # train-630        1890   val-216  216
    fear_path = os.path.join('%s/%s' % (emotion_path, 'fear'))           # train-270   495  1485   val-90   240
    happiness_path = os.path.join('%s/%s' % (emotion_path, 'happiness')) # train-720        2160   val-252  252
    neutral_path = os.path.join('%s/%s' % (emotion_path, 'neutral'))     # train-6318  1053 3159   val-354  354
    sadness_path = os.path.join('%s/%s' % (emotion_path, 'sadness'))     # train-288   528  1584   val-90   240
    surprise_path = os.path.join('%s/%s' % (emotion_path, 'surprise'))   # train-882        2646   val-306  306

    # 创建列表存储图片及图片对应的表情标签
    img = []
    label = []

    imglist = []
    labellist = []

    img_label_path = os.path.join('datasets/img_label_data', 'CK+_train_K_data.h5')
    if not os.path.exists(os.path.dirname(img_label_path)):
        os.makedirs(os.path.dirname(img_label_path))

    anger_imgs = os.listdir(anger_path)
    anger_imgs.sort()
    i = 0
    for anger_img in anger_imgs:
        i += 1
        img = cv2.imread(os.path.join(anger_path, anger_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # print('anger_img')
        # print(type(img))
        # print(img.shape)
        imglist.append(img)
        labellist.append(0)

        # I = skimage.io.imread(os.path.join(anger_path, anger_img))
        # # I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
        # # print('anger_img')
        # # print(type(I))
        # # print(I.shape)
        # img.append(I.tolist())
        # label.append(0)
        # # print(img)
        # # print(label)
    print('anger_img_num')
    print(i)

    contempt_imgs = os.listdir(contempt_path)
    contempt_imgs.sort()
    j = 0
    for contempt_img in contempt_imgs:
        j += 1
        img = cv2.imread(os.path.join(contempt_path, contempt_img),0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(1)
        # I = skimage.io.imread(os.path.join(contempt_path, contempt_img))
        # # print('contempt_img')
        # # print(type(I))
        # img.append(I.tolist())
        # label.append(1)
    print('contempt_img_num')
    print(j)

    disgust_imgs = os.listdir(disgust_path)
    disgust_imgs.sort()
    k = 0
    for disgust_img in disgust_imgs:
        k += 1
        img = cv2.imread(os.path.join(disgust_path, disgust_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(2)
        # I = skimage.io.imread(os.path.join(disgust_path, disgust_img))
        # img.append(I.tolist())
        # label.append(2)
    print('disgust_img_num')
    print(k)

    fear_imgs = os.listdir(fear_path)
    fear_imgs.sort()
    m = 0
    for fear_img in fear_imgs:
        m += 1
        img = cv2.imread(os.path.join(fear_path, fear_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(3)
        # I = skimage.io.imread(os.path.join(fear_path, fear_img))
        # img.append(I.tolist())
        # label.append(3)
    print('fear_img_num')
    print(m)

    happiness_imgs = os.listdir(happiness_path)
    happiness_imgs.sort()
    n = 0
    for happiness_img in happiness_imgs:
        n += 1
        img = cv2.imread(os.path.join(happiness_path, happiness_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(4)
        # I = skimage.io.imread(os.path.join(happiness_path, happiness_img))
        # img.append(I.tolist())
        # label.append(4)
    print('happiness_img_num')
    print(n)

    neutral_imgs = os.listdir(neutral_path)
    neutral_imgs.sort()
    a = 0
    for neutral_img in neutral_imgs:
        a += 1
        img = cv2.imread(os.path.join(neutral_path, neutral_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(5)
        # I = skimage.io.imread(os.path.join(neutral_path, neutral_img))
        # img.append(I.tolist())
        # label.append(5)
    print('neutral_img_num')
    print(a)

    sadness_imgs = os.listdir(sadness_path)
    sadness_imgs.sort()
    b = 0
    for sadness_img in sadness_imgs:
        b += 1
        img = cv2.imread(os.path.join(sadness_path, sadness_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imglist.append(img)
        labellist.append(6)
        # I = skimage.io.imread(os.path.join(sadness_path, sadness_img))
        # img.append(I.tolist())
        # label.append(6)
    print('sadness_img')
    print(b)

    surprise_imgs = os.listdir(surprise_path)
    surprise_imgs.sort()
    c = 0
    for surprise_img in surprise_imgs:
        c += 1
        img = cv2.imread(os.path.join(surprise_path, surprise_img), 0)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # print(img.shape)
        imglist.append(img)
        labellist.append(7)
        # I = skimage.io.imread(os.path.join(surprise_path, surprise_img))
        # img.append(I.tolist())
        # label.append(7)
    print('surprise_img_num')
    print(c)

    # print(np.shape(img))
    # print(np.shape(label))

    img_np = np.array(imglist)
    label_np = np.array(labellist)
    print(np.shape(img_np))
    print(np.shape(label_np))
    # img_np = np.concatenate(imglist)
    # label_np = np.concatenate(labellist)
    # length = len(labellist)
    # img_np.reshape(length, 3, 224, 224)


    datafile = h5py.File(img_label_path, 'w')
    datafile['img'] = img_np
    datafile['label'] = label_np
    # datafile.create_dataset("img", data=img)
    # datafile.create_dataset("label", dtype='int64', data=label)
    datafile.close()

    print('Save data finish!!!!!')




def main():
    emotion_label(CK_train_path)


if __name__ == '__main__':
    main()