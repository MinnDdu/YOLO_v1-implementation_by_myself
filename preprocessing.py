import os
import glob
import pandas as pd
import numpy as np
import shutil
import xmltodict
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# 1. 이미지 전처리
# 2. train/valid split
# 3. 이미지 라벨링
# 4. 데이터 증강

train_x_path = '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/train/images/'
val_x_path = '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/valid/images/'
test_x_path = '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/test/images/'

train_x_list = sorted(os.listdir(train_x_path))
val_x_list = sorted(os.listdir(val_x_path))
test_x_list = sorted(os.listdir(test_x_path))
# print(train_x.shape, val_x.shape, test_x.shape)

def png2matrix(path1, path2, path3, list1, list2, list3): # return matricies of train_x, val_x, and test_x
    train_x_mat, val_x_mat, test_x_mat = [], [], []
    for i in range(len(list1)):
        matrix = np.array(Image.open(train_x_path + train_x_list[i]).resize((224, 224)))
        train_x_mat.append(matrix)
    for i in range(len(list2)):
        matrix = np.array(Image.open(val_x_path + val_x_list[i]).resize((224, 224)))
        val_x_mat.append(matrix)
    for i in range(len(list3)):
        matrix = np.array(Image.open(test_x_path + test_x_list[i]).resize((224, 224)))
        test_x_mat.append(matrix)
    return train_x_mat, val_x_mat, test_x_mat

train_x_mat, val_x_mat, test_x_mat = png2matrix(train_x_path, val_x_path, test_x_path, train_x_list, val_x_list, test_x_list)
print(train_x_mat[0].shape)
print(val_x_mat[0].shape)
print(test_x_mat[0].shape)
# 각각의 리스트에 (224, 224, 3) 형태의 행렬이 저장된 모습!


# 데이터셋에 어떤 종류의 class들이 있는지 리스트 만들기 - todo


# 데이터셋 라벨링 - todo

