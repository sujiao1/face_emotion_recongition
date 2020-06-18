import os, shutil
import cv2

original_dir = 'datasets/CK+_imgs'
target_dir = 'datasets/CK+_renew/CK+_re_merge/test'


def emotion_resize_file(original_dir, target_dir):
    landmark_norm_tails = os.listdir(original_dir)
    print(landmark_norm_tails)
    for landmark_norm_tail in landmark_norm_tails:
        if landmark_norm_tail == 'test':
            landmark_norm_tail_path = os.path.join('%s/%s' % (original_dir, landmark_norm_tail))
            print(landmark_norm_tail_path)
            emotions = os.listdir(landmark_norm_tail_path)
            print(emotions)
            for emotion in emotions:
                if emotion is not None:
                    emotion_path = os.path.join('%s/%s' % (landmark_norm_tail_path, emotion))
                    print(emotion_path)
                    imgs = os.listdir(emotion_path)
                    print(imgs)
                    target_emotion_dir = os.path.join('%s/%s' % (target_dir, emotion))
                    if not os.path.exists(target_emotion_dir):
                        os.makedirs(target_emotion_dir)
                    i = 0
                    for img in imgs:
                        i += 1
                        img_path = os.path.join('%s/%s' % (emotion_path, img))
                        image = cv2.imread(img_path, 0)
                        resize_img = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
                        # shutil.copy(img_path, os.path.join(target_emotion_dir, img))
                        cv2.imwrite(target_dir + '/' + emotion + '/' + img, resize_img)
                    print(i)



def main():
    emotion_resize_file(original_dir, target_dir)

if __name__ == '__main__':
    main()