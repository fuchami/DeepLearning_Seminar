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

import matplotlib.pyplot as plt

# 各パラメータの設定
batch_size = 32
num_classes = 256 
epochs = 30 
img_shape = (128,128)

def plot_history(history):
        # print(history.history.keys())

        # 精度の履歴をプロット
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['acc', 'val_acc'], loc='lower right')
        plt.show()

        # 損失の履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        plt.show()

# caltech256 のロード
# トレーニングデータには画像を水増したくさんするよ
train_datagen = ImageDataGenerator(
       rescale=1.0 / 255,
       shear_range=0.2,
       zoom_range=0.2,
       height_shift_range=0.1,
       width_shift_range=0.1,
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
fine_model = InceptionV3(include_top=False, weights='imagenet',input_tensor=input_tensor)

# ひとまず全結合層のモデルを作成（後にinveption-v3と結合させる）
fc_model = Sequential()
fc_model.add(Flatten(input_shape=fine_model.output_shape[1:]))

fc_model.add(Dense(512, activation='relu'))
fc_model.add(Dropout(0.25))
fc_model.add(Dense(num_classes, activation='softmax'))

# 上の2つのモデルを結合
model = Model(input=fine_model.input, output=fc_model(fine_model.output))

# 最後のconv層の直前までの層をfreeze
for layer in fine_model.layers[:249]:
    layer.trainable = False

model.summary()

# 学習パラメータの設定
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])

# 学習開始
history = model.fit_generator(train_generator,
        epochs = epochs,
        steps_per_epoch = 200,
        validation_data=test_generator,
        validation_steps= 100)

plot_history(history)