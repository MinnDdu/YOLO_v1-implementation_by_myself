import os
import glob
import pandas as pd
import numpy as np
import shutil
import xmltodict
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import json

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
        matrix = np.array(Image.open(path1 + list1[i]).resize((224, 224)))
        train_x_mat.append(matrix)
    for i in range(len(list2)):
        matrix = np.array(Image.open(path2 + list2[i]).resize((224, 224)))
        val_x_mat.append(matrix)
    for i in range(len(list3)):
        matrix = np.array(Image.open(path3 + list3[i]).resize((224, 224)))
        test_x_mat.append(matrix)
    # divide into 255.0 
    train_x_mat = np.divide(train_x_mat, 255.0)
    val_x_mat = np.divide(val_x_mat, 255.0)
    test_x_mat = np.divide(test_x_mat, 255.0)

    return train_x_mat, val_x_mat, test_x_mat

train_x_mat, val_x_mat, test_x_mat = png2matrix(train_x_path, val_x_path, test_x_path, train_x_list, val_x_list, test_x_list)
# print(train_x_mat[0].shape)
# print(val_x_mat[0].shape)
# print(test_x_mat[0].shape)
# 각각의 리스트에 (224, 224, 3) 형태의 행렬이 저장된 모습!


# 데이터셋에 어떤 종류의 class들이 있는지 리스트 만들기 - todo
classes = []
def get_classes_from_xml(path, lst):
    xml_list = sorted(os.listdir(path))
    for i in range(len(xml_list)):
        with open(path+'/'+xml_list[i], 'r') as file:
            xml = file.read()
            dict_ = xmltodict.parse(xml)
            # print(xml_list[i])

            obj = dict_['annotation']['object']
            if type(obj) == list: # 사진에 오브젝트가 여러개일때
                for i in obj:
                    lst.append(i['name'])
            elif type(obj) == dict: # 사진에 오브젝트가 하나일때
                lst.append(obj['name'])
            else:
                print(xml_list[i])


# get_classes_from_xml('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/train/xmls', classes)
# get_classes_from_xml('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/valid/xmls', classes)
# get_classes_from_xml('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/test/xmls', classes)

classes = sorted(list({'bus', 'horse', 'cow', 'car', 'pottedplant', 'diningtable', 'aeroplane', 'person', 'bottle', 'bird', 'cat', 'chair', 'boat', 'sheep', 'bicycle', 'tvmonitor', 'dog', 'train', 'sofa', 'motorbike'}))
classes_dict = {}
for i in range(len(classes)):
    classes_dict[classes[i]] = i


# 데이터셋 라벨링
def get_label_position(xml_path):
    xml_list = sorted(os.listdir(xml_path))

    # 모델의 final output tensor => (7x7x30) (SxSx(B*5+C)) - S=7, B=2, C=20
    # 그런데 라벨데이터에서는 객체 1개 (진짜 객체)의 박스만 존재하므로 B=1 => 7x7x(1*5+20)꼴로 세팅
    label = np.zeros((7, 7, 25)) 

    for i in range(len(xml_list)):
        file = open(xml_path + '/' + xml_list[i], 'r')
        xml = file.read()
        dict_ = xmltodict.parse(xml)
        # original image size
        img_size_width = float(dict_['annotation']['size']['width'])
        img_size_height = float(dict_['annotation']['size']['height'])
        img_size_depth = float(dict_['annotation']['size']['depth'])

        if type(dict_['annotation']['object']) == list: # 객체 여러개
            obj_lst = dict_['annotation']['object']
            for obj in obj_lst:
                class_index = classes_dict[obj['name']]
                # 224 x 224 size coordinate
                xmin = (float(obj['bndbox']['xmin']) / img_size_width) * 224.0 # 0.0 ~ 224.0
                ymin = (float(obj['bndbox']['ymin']) / img_size_height) * 224.0 # 0.0 ~ 224.0
                xmax = (float(obj['bndbox']['xmax']) / img_size_width) * 224.0 # 0.0 ~ 224.0
                ymax = (float(obj['bndbox']['ymax']) / img_size_height) * 224.0 # 0.0 ~ 224.0
                # Each bounding box consists of 5 predictions - x,y,w,h,confidence
                x = (xmax + xmin) / 2.0 # bndbox 중심의 x좌표
                y = (ymax + ymin) / 2.0 # bndbox 중심의 y좌표
                w = xmax - xmin # bndbox의 너비
                h = ymax - ymin # bndbox의 높이
                # 7x7 (SxS)의 cell grid에서 어떤 cell들이 이 객체를 포함하는지!
                # 아직 좀더 이해필요... (이 코드는 객체전체가 아니라 바운딩 박스 중심점만 고려하는거 아닌가..?)
                x_cell = int(x/(224/7)) # 0 ~ 6 (7 cells)
                y_cell = int(y/(224/7)) # 0 ~ 6 (7 cells)

                label[y_cell][x_cell][class_index] = 1.0 # 0~19 index의 객체 중 탐지된 객체
                label[y_cell][x_cell][20] = (x - (x_cell * 32)) / 32.0 # index 20 - x
                label[y_cell][x_cell][21] = (y - (y_cell * 32)) / 32.0 # index 21 - y
                label[y_cell][x_cell][22] = w / 224.0 # index 22 - w - 0.0 ~ 224.0
                label[y_cell][x_cell][23] = h / 224.0 # index 23 - h - 0.0 ~ 224.0
        else: # 객체 1개
            obj = dict_['annotation']['object']
            class_index = classes_dict[obj['name']]
            # 224 x 224 size coordinate
            xmin = (float(obj['bndbox']['xmin']) / img_size_width) * 224.0 # 0.0 ~ 224.0
            ymin = (float(obj['bndbox']['ymin']) / img_size_height) * 224.0 # 0.0 ~ 224.0
            xmax = (float(obj['bndbox']['xmax']) / img_size_width) * 224.0 # 0.0 ~ 224.0
            ymax = (float(obj['bndbox']['ymax']) / img_size_height) * 224.0 # 0.0 ~ 224.0
            # Each bounding box consists of 5 predictions - x,y,w,h,confidence
            x = (xmax + xmin) / 2.0 # bndbox 중심의 x좌표
            y = (ymax + ymin) / 2.0 # bndbox 중심의 y좌표
            w = xmax - xmin # bndbox의 너비
            h = ymax - ymin # bndbox의 높이
            # 7x7 (SxS)의 cell grid에서 어떤 cell들이 이 객체를 포함하는지!
            # 아직 좀더 이해필요... (이 코드는 객체전체가 아니라 바운딩 박스 중심점만 고려하는거 아닌가..?)
            x_cell = int(x/(224/7)) # 0 ~ 6 (7 cells)
            y_cell = int(y/(224/7)) # 0 ~ 6 (7 cells)

            label[y_cell][x_cell][class_index] = 1.0 # 0~19 index의 객체 중 탐지된 객체
            label[y_cell][x_cell][20] = (x - (x_cell * 32)) / 32.0 # index 20 - x
            label[y_cell][x_cell][21] = (y - (y_cell * 32)) / 32.0 # index 21 - y
            label[y_cell][x_cell][22] = w / 224.0 # index 22 - w - 0.0 ~ 224.0
            label[y_cell][x_cell][23] = h / 224.0 # index 23 - h - 0.0 ~ 224.0
    
    return label


train_x_label_mat = get_label_position('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/train/xmls')
val_x_label_mat = get_label_position('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/valid/xmls')
test_x_label_mat = get_label_position('/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/test/xmls')
print(train_x_label_mat)
print(train_x_label_mat.shape)
