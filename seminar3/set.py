# coding: utf-8

"""
画像データを訓練データと検証データに振り分ける前処理PG

こんな感じにしたい
data
├── train
│   ├── cats   cat0001.jpg - cat1000.jpg
│   └── dogs   dog0001.jpg - dog1000.jpg
└── validation
    ├── cats   cat0001.jpg - cat0400.jpg
    └── dogs   dog0001.jpg - dog0400.jpg


"""
# 各ライブラリのインポテンツ
import os, sys
import glob
import random
import shutil

# もともとのCaltech101データセットのソースディレクトリ
source_dir = './101_ObjectCategories'
# 訓練データ，テストデータのコピー先ディレクトリ
train_dir  = './data/train'
valid_dir  = './data/validation'

# テストデータサイズ
TEST_SIZE = 10

# クラス名を取得
class_path_list = os.listdir(source_dir)
print (class_path_list)

for class_path in class_path_list:
    print (class_path)

    # クラス内の画像リストの取得
    img_list = os.listdir(source_dir + "/" + class_path)
    # 取得した画像リストをランダムにシャッフル
    random.shuffle(img_list)

    # 移動先ディレクトがなければ作成
    # 訓練データ格納ディレクトリ
    if not os.path.exists(train_dir + "/" + class_path):
        os.makedirs(train_dir + "/"+ class_path)

    # テストデータ格納ディレクトリ
    if not os.path.exists(valid_dir + "/" + class_path):
        os.makedirs(valid_dir + "/"+ class_path)

    # ひとまず全てを訓練データに
    for i in range(len(image_list)):
        shutil.copyfile("%s/%s/%s" % (source_dir, class_path, img_list[i]),
                            '%s/%s/img%04d.jpg' % (train_dir, class_path,i))

    # 訓練データディレクトリから画像リストを再取得
    image_list = os.listdir(train_dir + "/" + class_path)
    # 取得した画像リストをランダムにシャッフル
    random.shuffle(img_list)

    # テストデータを訓練データディレクトリからコピーして格納
    for i in range(TEST_SIZE):
        os.rename("%s/%s/%s" % (train_dir, class_path, img_list[i]),
                            '%s/%s/img%04d.jpg' % (valid_dir, class_path,i))
