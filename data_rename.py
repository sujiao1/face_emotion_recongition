import os

data_path = 'datasets/CK+_renew/CK+_resize/val'


def data_rename(data_path):
    landmark_norm_tails = os.listdir(data_path)
    print(landmark_norm_tails)
    for landmark_norm_tail in landmark_norm_tails:
        landmark_norm_tail_path = os.path.join('%s/%s' % (data_path, landmark_norm_tail))
        print(landmark_norm_tail_path)
        emotions = os.listdir(landmark_norm_tail_path)
        for emotion in emotions:
            emotion_path = os.path.join('%s/%s' % (landmark_norm_tail_path, emotion))
            print(emotion_path)
            imgs = os.listdir(emotion_path)
            print(imgs)
            for i, img in enumerate(imgs):
                img_path = os.path.join('%s/%s' % (emotion_path, img))
                print(img_path)
                src = os.path.join(os.path.abspath(emotion_path), img)
                dst = os.path.join(os.path.abspath(emotion_path),
                                   'CK+_val' + landmark_norm_tail + emotion + str(i) + '.jpg')
                os.rename(src, dst)

def main():
    data_rename(data_path)


if __name__ == '__main__':
    main()