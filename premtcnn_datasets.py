# -*- coding: utf-8 -*-
import sys
import csv
import math
import os
from PIL import Image

imgs_root = "datasets/fer2013imgs/train"

def predatasets(images_root):
    expression_dirlist = os.listdir(images_root)
    print(expression_dirlist)
    for i in expression_dirlist:
        img_path = os.listdir(images_root + "/" + i)
        print(img_path)



def main():
    predatasets(imgs_root)


if __name__ == '__main__':
    main()