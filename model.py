import tensorflow as tf
import numpy as np
import os

# models - Backbone + Head
# Backbone part - imagenet
yolo_model = tf.keras.Sequential()

max_num = len(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers)

for i in range(0, max_num):
    yolo_model.add(tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).layers[i])

for layer in yolo_model.layers:
    layer.trainable = False
    if hasattr(layer, 'activation'):
        layer.activation = tf.keras.layers.LeakyReLU(alpha=0.1)
# -----------------------------------------------------------------
# Head part
yolo_model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same'))
yolo_model.add(tf.keras.layers.LeakyReLU(0.1)) # conv의 activation func 역할
yolo_model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same'))
yolo_model.add(tf.keras.layers.LeakyReLU(0.1)) # conv의 activation func 역할

# yolo_model.add(tf.keras.layers.MaxPool2D((2,2)))

yolo_model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same'))
yolo_model.add(tf.keras.layers.LeakyReLU(0.1)) # conv의 activation func 역할
yolo_model.add(tf.keras.layers.Conv2D(1024, (3,3), padding='same'))
yolo_model.add(tf.keras.layers.LeakyReLU(0.1)) # conv의 activation func 역할
# Head - linear part
yolo_model.add(tf.keras.layers.Flatten())
yolo_model.add(tf.keras.layers.Dense(4096))
yolo_model.add(tf.keras.layers.LeakyReLU(0.1)) # dense의 activation func 역할
yolo_model.add(tf.keras.layers.Dropout(0.5))
yolo_model.add(tf.keras.layers.Dense(7*7*30))
yolo_model.add(tf.keras.layers.Reshape((7, 7, 30)))

print(yolo_model.summary())
tf.keras.utils.plot_model(yolo_model, './yolo.png', True, True)


# note - 디테일한 파라미터들 레이어에 넣어줘야함! 다음번에 넣기