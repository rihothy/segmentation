import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import random
from U_Net import*
import numpy as np

n = 512
l = list(range(1034))
random.shuffle(l)

def train_gen():
    global l, n
    while True:
        m_l = l[:900]
        random.shuffle(m_l)
        for i in m_l:
            x = cv2.imread('train_set/x/'+str(i)+'.png')
            y = cv2.imread('train_set/y/'+str(i)+'.png', 0)

            if random.random() < 0.5:
                x = cv2.flip(x, 0)
                y = cv2.flip(y, 0)
            if random.random() < 0.5:
                x = cv2.flip(x, 1)
                y = cv2.flip(y, 1)

            x = x[np.newaxis, :, :, :].astype('float32') / 255
            y = keras.utils.to_categorical(y, 5)
            y = y.reshape(1, n, n, 5)

            yield (x, y)

def test_gen():
    global l, n
    while True:
        for i in l[900:]:
            x = cv2.imread('train_set/x/'+str(i)+'.png')
            y = cv2.imread('train_set/y/'+str(i)+'.png', 0)

            if random.random() < 0.5:
                x = cv2.flip(x, 0)
                y = cv2.flip(y, 0)
            if random.random() < 0.5:
                x = cv2.flip(x, 1)
                y = cv2.flip(y, 1)

            x = x[np.newaxis, :, :, :].astype('float32') / 255
            y = keras.utils.to_categorical(y, 5)
            y = y.reshape(1, n, n, 5)

            yield (x, y)


model = U_Net()
print(model.summary())

check_point = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_binary_accuracy', save_best_only=True, mode='max')
model.fit_generator(generator=train_gen(), steps_per_epoch=934, validation_data=test_gen(), validation_steps=100, epochs=512, callbacks=[check_point])