import os
import glob
import pandas as pd
import numpy as np
import shutil
import xmltodict
import tensorflow as tf


# ordinary_image_path = '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/VOC2007/JPEGImages'
# ordinary_xml_path = '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/VOC2007/Annotations'
# ordinary_image_path = sorted(os.listdir(ordinary_image_path))
# ordinary_xml_path = sorted(os.listdir(ordinary_xml_path))

# # train/valid/test split -> shutil 라이브러리 이용해 직접하기... 다음엔 sklearn 라이브러리 이용해서 해보자...
# # 비율은 대략 6:2:2 정도로

# def organize_dataset():
#     # train set (4000)
#     for i in range(0, 4000):
#         shutil.move(ordinary_image_path + '/' + ordinary_image_path[i], '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/train/images/' + ordinary_image_path[i])
#         shutil.move(ordinary_xml_path + '/' + ordinary_xml_path[i], '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/train/xmls/' + ordinary_xml_path[i])
#     # valid set (about 1000)
#     shutil.move(ordinary_image_path + '/', '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/valid/images/')
#     shutil.move(ordinary_xml_path + '/', '/Users/minsoo/Desktop/CS_practice/Paper Implementations/YOLO_v1/YOLO_v1-implementation_by_myself/dataset/valid/xmls/')

# ti = sorted(os.listdir('dataset/train/images'))
# tx = sorted(os.listdir('dataset/train/xmls'))

# for i in range(4000-1, 4000-1-800, -1):
#     shutil.move('dataset/train/images/' + ti[i], 'dataset/valid/images/' + ti[i])
#     shutil.move('dataset/train/xmls/' + tx[i], 'dataset/valid/xmls/' + tx[i])
