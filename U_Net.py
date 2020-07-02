import keras
from keras import backend as K
import tensorflow as tf

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


def multi_category_focal_loss1(alpha, gamma=2.0):
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed


def U_Net(input_size=(None, None, 3)):
    inputs = keras.Input(input_size)

    conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
    conv1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
    drop1 = keras.layers.Dropout(0.5)(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
    conv2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
    drop2 = keras.layers.Dropout(0.5)(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
    conv3 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
    drop3 = keras.layers.Dropout(0.5)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
    conv4 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
    conv5 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop5)

    conv6 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(pool5)
    conv6 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv6)
    drop6 = keras.layers.Dropout(0.5)(conv6)

    upsp7 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size=(2,2))(drop6))
    merg7 = keras.layers.concatenate([conv5, upsp7], axis=3)
    conv7 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merg7)
    conv7 = keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)

    upsp8 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size=(2,2))(conv7))
    merg8 = keras.layers.concatenate([conv4, upsp8], axis=3)
    conv8 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merg8)
    conv8 = keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)

    upsp9 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size=(2,2))(conv8))
    merg9 = keras.layers.concatenate([conv3, upsp9], axis=3)
    conv9 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merg9)
    conv9 = keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)

    upsp10 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size=(2,2))(conv9))
    merg10 = keras.layers.concatenate([conv2, upsp10], axis=3)
    conv10 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merg10)
    conv10 = keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv10)

    upsp11 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size=(2,2))(conv10))
    merg11 = keras.layers.concatenate([conv1, upsp11], axis=3)
    conv11 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merg11)
    conv11 = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv11)

    conv12 = keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv11)
    conv12 = keras.layers.Conv2D(5, 1, activation='sigmoid')(conv12)

    model = keras.models.Model(inputs=inputs, outputs=conv12)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=[multi_category_focal_loss1(alpha=[[0.57], [0.38], [0.52], [1.65], [1.88]])], metrics=['binary_accuracy'])

    return model