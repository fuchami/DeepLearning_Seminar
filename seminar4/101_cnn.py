# coding: utf-8

# 手書き数字画像データセットMNISTを使って画像認識
# CNNの実装


# 各ライブラリのインポート
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Flatten
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, Maxpooling2D

# 各パラメータの設定
batch_size = 32 
num_classes = 101 
epochs = 30      

# caltech101 のロード

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale= 1.0/ 255)

train_generator = train_datagen.flow_from_directory(
        "./data/train",
        target_size=(32,32),
        batch_size= batch_size,
        color_mode = "rgb",
        class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
        "./data/validation",
        target_size=(32,32),
        batch_size= batch_size,
        color_mode = "rbg",
        class_mode = "categorical")

"""
 畳み込みニューラルネットワーク:CNN の構築
 kerasではモデルに層をどんどん積み重ねるように記述する
"""
model = Sequential()

# 2層の畳み込み 引数は順に，
# フィルター数，フィルターサイズ，パディング(valid or same),入力サイズ(縦，横，チャンネル数) 
model.add(Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))

# マックスプーリング
# プーリングは基本的に数回畳み込んだ後に行われる
model.add(Maxpooling2D(pool_size=(2,2)))
# ドロップアウト
# 学習時にニューロンをランダムに無効にさせてパラメータ更新を行う
# 自由度を強制的に小さくして汎用性能を上げ，過学習を防ぐことが出来る
# ここでは25%のニューロンを無効に
model.add(Dropout(0.25))

# どんどん積んでくよ〜
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# すべてのニューロンを全結合させてく(今までやった普通のニューラルネットにさせる)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 最後はクラス数だけのニューロンを設定．softmaxで確率を出力
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

# 学習パラメータの設定
model.compile(loss='categorical_crossentropy',
            optimizer=SGD(),
            metrics=['accuracy'])

# 学習開始
model.fit_generator(train_generator,
        epochs = epochs,
        steps_per_epoch = 100,
        validation_data=test_generator,
        validation_steps= 80)

# テストデータでモデルのスコアを算出
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])
