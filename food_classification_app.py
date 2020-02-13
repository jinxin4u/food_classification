import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf

import AccuracyHistory
from numpy.random import randn

import pathlib
import json
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 라벨링 및 트레이닝,테스트 분리
data_dir = "D:/korean_food/food"
data_dir = pathlib.Path(data_dir)

label_key = os.listdir(data_dir)
label_names = {}

for idx, value  in enumerate(label_key):
    label_names[value] = idx

all_images = list(data_dir.glob('*/*'))
all_images = [ str(path) for path in all_images ]
print(all_images)
np.random.shuffle(all_images)

all_labels = [ label_names[ pathlib.Path(path).parent.name ] for path in all_images ]
data_size = len( all_images )
train_test_split = (int)( data_size * 0.2 )

x_train = all_images[ train_test_split: ]
x_test = all_images[ :train_test_split ]
y_train = all_labels[ train_test_split: ]
y_test = all_labels[ :train_test_split ]



# 데이터 전처리 및 데이터셋

IMG_SIZE = 100
BATCH_SIZE = 10
EPOCH = 1

def data_preprocessing( x, y ):
    image = tf.io.read_file( x )
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1 # -1~1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, y

def data_setting( x, y ):
    ds = tf.data.Dataset.from_tensor_slices( (x,y) ) # 데이터셋 생성
    ds = ds.map( data_preprocessing )

    ds = ds.shuffle( buffer_size = data_size )
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch( buffer_size = AUTOTUNE)
    return ds

train_ds = data_setting( x_train, y_train )
validation_ds = data_setting( x_test, y_test )


# 모델링

model = Sequential()

model.add( Conv2D(100, 4, 4, activation='relu', input_shape=(100, 100, 3)) )
model.add( Conv2D(100, 4, 4, activation='relu'))
model.add( MaxPooling2D( pool_size=(2,2) ))
model.add( Dropout(0.5) )
model.add( Flatten() )

model.add( Dense(200, activation='relu') )
model.add( Dropout(0.5) )
model.add( Dense(150, activation='softmax'))


history = AccuracyHistory()

model.compile( loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta,
               metrics=['accuracy'])

history = model.fit(train_ds, epochs=EPOCH, validation_data=validation_ds,
                    verbose=1, callbacks=[history])

score = model.evaluate(validation_ds, verbose=0)