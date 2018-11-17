# coding: utf-8

# 手書き数字画像データセットMNISTを使って画像認識
# 多層パーセプトロンの実装

# 各ライブラリのインポート
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD,Adam

# 各パラメータの設定
batch_size = 128  #SGDではバッチサイズごとに勾配を求め最適化を行う
num_classes = 10  #クラス数(0~9までの数字なので10クラス分類)
epochs = 20       #どんくらい学習を行うかの大まかな回数

# MNIST データセットの読み込み
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

"""
機械学習は基本的にデータを
訓練データ(Train data)とテストデータ(validatio data)に分ける
出来上がったモデルに対して，未知のデータであるテストデータを入力し，性能を図る

訓練データで精度が良いがテストデータで精度が悪い場合は
訓練データにオーバーフィッティング(過学習)しており，汎用的なモデルではない
"""

# 訓練データ：28x28 (=784次元)の数字画像が 60,000枚
# テストデータ：28x28(=784次元)の数字画像が 10,000枚
"""
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
"""
# => 省略してこうかける
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float32') / 255

# 正解ラベルをワンホットベクターに変換
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test  = keras.utils.to_categorical(Y_test, num_classes)

"""
 多層パーセプトロン;MLP の構築
 kerasではモデルに層をどんどん積み重ねるように記述する
"""
model = Sequential()

# 隠れ層1:入力が784次元, ノード数512, 活性化関数にシグモイド関数
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('sigmoid'))

# 隠れ層2:ノード数256, 活性化関数にシグモイド関数
model.add(Dense(256))
model.add(Activation('sigmoid'))

# 出力層: ノード数10(10クラス分類なので) 活性化関数にソフトマックス関数
model.add(Dense(10))
model.add(Activation('softmax'))

# モデル全体を出力
model.summary()

# 学習パラメータの設定
# - 目的関数：categorical_crossentropy
# - 最適化アルゴリズム：rmsprop
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(),
            metrics=['accuracy'])

# 学習開始
model.fit(X_train, Y_train,
        batch_size=batch_size,
        nb_epoch = epochs,
        verbose=1,
        validation_data=(X_test, Y_test))

# テストデータでモデルのスコアを算出
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

#バックエンドをインポート
from keras.backend import tensorflow_backend as backend
#処理終了時に下記をコール
backend.clear_session()