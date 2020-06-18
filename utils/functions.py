from config import *
import os
import cv2

# 创建模型文件，实验文件，及训练后的模型文件
def create_train_dir(params):
    """
    Create folder used in training, folder hierarchy:
        current folder--exp_folder
                       |
                       --summaries
                       --checkpoints
        the exp_folder is named by model_name + dataset_name
    """
    experiment = 'models_' + params.model + '_' + params.dataset
    # exp_dir = os.path.join('%s/%s' % (params.root, experiment))
    exp_dir = os.path.join(os.getcwd(), params.root, experiment)
    print(exp_dir)
    # if not os.path.exists(exp_dir):
    #     print('11111111111')
    #     os.mkdir(exp_dir)
    summary_dir = os.path.join('%s/%s' % (exp_dir, 'summaries/'))
    print(summary_dir)
    checkpoint_dir = os.path.join('%s/%s' % (exp_dir, 'checkpoints/'))
    print(checkpoint_dir)

    dirs = [exp_dir, summary_dir, checkpoint_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            print('111')
            os.makedirs(dir)

    return summary_dir, checkpoint_dir


def print_config(params):
    for name, value in sorted(vars(params).items()):
        print('\t%-20s:%s' % (name, str(value)))
    print('')


def draw_text_pos(position, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = position[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


# def main():
#     params = Params()
#     s, c = create_train_dir(params)
#     # if not os.path.exists(s):
#     #     os.mkdir(s)
#     # if os.path.exists(c):
#     #     os.mkdir(c)
#     # print_config(params)
#
# if __name__ == '__main__':
#     main()