# coding: utf-8

# Caltech256を使って画像認識
# CNNの実装

# 各ライブラリのインポート
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Activation 
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

# 各パラメータの設定
batch_size = 32 
num_classes = 101 
epochs = 30
img_shape = (128,128)

# caltech256 のロード
train_datagen = ImageDataGenerator(
       rescale=1.0 / 255,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale= 1.0/ 255)

train_generator = train_datagen.flow_from_directory(
        "./data/train",
        target_size=img_shape,
        batch_size= batch_size,
        color_mode = "rgb",
        class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
        "./data/validation",
        target_size=img_shape,
        batch_size= batch_size,
        color_mode = "rgb",
        class_mode = "categorical")

"""
 Inception-v3をFine Tuningしてcaltech101のクラス分類用にフィッティングさせる
"""

# input tensorを定義（これしないとエラーになるっぽい？）
input_tensor = Input(shape=(128,128,3))
# 全結合層(top)はFalseで除外，imagenetで学習させた重みを読み込んでv3を構築
fine_model = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)

# ひとまず全結合層のモデルを作成（後にinveption-v3と結合させる）
fc_model = Sequential()
fc_model.add(Flatten(input_shape=fine_model.output_shape[1:]))
fc_model.add(Dense(256, activation='relu'))
fc_model.add(Dropout(0.5))
fc_model.add(Dense(num_classes, activation='softmax'))

# 上の2つのモデルを結合
model = Model(input=fine_model.input, output=fc_model(fine_model.output))

model.summary()

# 学習パラメータの設定
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])

# 学習開始
model.fit_generator(train_generator,
        epochs = epochs,
        steps_per_epoch = 100,
        validation_data=test_generator,
        validation_steps= 80)

