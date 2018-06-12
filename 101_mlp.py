# coding: utf-8

# 手書き数字画像データセットMNISTを使って画像認識
# 多層パーセプトロンの実装

# 各ライブラリのインポート
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import array_to_img, load_img

# 必要なライブラリを追加
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 各パラメータの設定
batch_size = 32 #バッチサイズ
num_classes = 101 #クラス数
epochs = 30 #エポック数
input_shape=(32,32,1) #入力する画像の形 (row, col, channel)

# 精度とLossの学習経過を描画する関数
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

        # Lossの履歴をプロット
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        plt.show()


#caltech101 datasets load
# 訓練データとテストデータを生成するジェネレーターを作成

trian_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

trian_generator = trian_datagen.flow_from_directory(
        "./data/train",
        target_size=(32,32),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
        "./data/validation",
        target_size=(32,32),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode="categorical")

"""
 多層パーセプトロン;MLP の構築
 kerasではモデルに層をどんどん積み重ねるように記述する
"""

model = Sequential()

# 隠れ層1:ノード数128, 活性化関数にシグモイド関数
model.add(Dense(128, input_shape=input_shape))
model.add(Activation('sigmoid'))

# 隠れ層2:ノード数64,活性化関数にシグモイド関数
model.add(Dense(64))
model.add(Activation('sigmoid'))

# 全結合層に変換(2次元→1次元の入力に)
model.add(Flatten())

# 出力層: ノード数101(101クラス分類なので) 活性化関数にソフトマックス関数
model.add(Dense(101))
model.add(Activation('softmax'))

# モデル全体を出力
model.summary()

# 学習パラメータの設定
# - 目的関数：categorical_crossentropy
# - 最適化アルゴリズム：SGD
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(),
            metrics=['accuracy'])
# 学習開始
# パラメータは適当に決めて
history = model.fit_generator(trian_generator,
        epochs = epochs,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=80)

# 学習経過を描画
plot_history(history)
