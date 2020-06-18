##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil

# 源图片文件夹路径
original_dir = 'datasets/CK+_renew/CK+_renorm/val'

# 目标文件夹
target_dir = 'datasets/CK+_renew/CK+_re_merge/val/'
target_train_dir = os.path.join(target_dir, 'train_K_data')
target_val_dir = os.path.join(target_dir, 'val')
target_test_dir = os.path.join(target_dir, 'test')

def moveFile(original_dir, target_dir):
    expression_Dir = os.listdir(original_dir)  # 取图片的原始路径
    print(expression_Dir)
    for expression in expression_Dir:
        # if expression != 'neutral':
        imgs_path = os.path.join('%s/%s/' % (original_dir, expression))
        print(imgs_path)
        images = os.listdir(imgs_path)
        filenumber = len(images)
        print(filenumber)
        rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
        print(picknumber)
        sample = random.sample(images, picknumber)  # 随机选取picknumber数量的样本图片
        print(sample)

        target_expression_dir = os.path.join("%s/%s/" % (target_dir, expression))
        if not os.path.exists(target_expression_dir):
            os.mkdir(target_expression_dir)

        for name in sample:
            # shutil.move(imgs_path + name, target_expression_dir + name)
            shutil.copy(imgs_path + name, target_expression_dir + name)
    return


if __name__ == '__main__':
    # fileDir = "./source/"  # 源图片文件夹路径
    # tarDir = './result/'  # 移动到新的文件夹路径
    moveFile(original_dir, target_dir)
















