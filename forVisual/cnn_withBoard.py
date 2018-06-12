# coding: utf-8

"""
kerasでtensorboardを使ってみようよ

　　　　　　　　 　／＼
　　　　　. ∵ .／　 .／|
　　 　 _, ,_ﾟ ∴＼／／
　　 (ﾉﾟДﾟ)ﾉ　　 |／
　　/　　/
￣￣￣￣￣￣ 
"""

import keras
from  keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils, plot_model

import tensorflow as tf
from keras.callbacks import TensorBoard


# ハイパラ定義
batch_size = 4
input_shape = (32,32,3)
num_classes = 10
epochs = 1
optimizers = [RMSprop(), SGD(), Adam()]

def build_cnn(input_shape, num_classes):
    with tf.name_scope('CNN_model') as scope:
            
            model = Sequential()
            with tf.name_scope('conv_1') as scope:
                model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
            with tf.name_scope('conv_2') as scope:
                model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
            with tf.name_scope('maxpool') as scope:
                model.add(MaxPooling2D(pool_size=(2,2)))
            

            with tf.name_scope('conv_3') as scope:
                model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
            with tf.name_scope('maxpool') as scope:
                model.add(MaxPooling2D(pool_size=(2,2)))

            with tf.name_scope('flatten') as scope:
                model.add(Flatten())
            
            with tf.name_scope('dense') as scope:
                model.add(Dense(1024, activation='relu'))

            with tf.name_scope('dropout') as scope:
                model.add(Dropout(0.25))

            with tf.name_scope('dense') as scope:
                model.add(Dense(num_classes, activation='softmax'))
            
            return model

def train_model(model, opt):
    
    log_dir = './log/optimizer={}'.format(opt)
    tb_cb = TensorBoard(log_dir=log_dir,
            histogram_freq=1,
            write_grads=True,
            write_graph=True,
            write_images=1)
    
    model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # 学習履歴を取得
    history = model.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, Y_test),
                callbacks=[tb_cb])
    

# load cifar 10
(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()

# いつもの如くデータを正規化
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# ラベルはOne-hotベクトル表現に
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

# first build model
model = build_cnn(input_shape, num_classes)

# initial weights
model.save_weights("initial_weights.hdf5")


for opt in optimizers:
    
    model.load_weights("initial_weights.hdf5")
    train_model(model, opt)
    


    