import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import random
from U_Net import*
import numpy as np

model = U_Net()
model.load_weights('model.h5')

print(model.summary())

img = cv2.imread('test.png')
img = img[:img.shape[0]//32*32, :img.shape[1]//32*32]
img = img[np.newaxis, :, :, :].astype('float32') / 255

result = model.predict(img)

result = result.squeeze()
result = np.argmax(result, axis=2)
result = np.uint8(result) * 63

cv2.imwrite('result.png', result)