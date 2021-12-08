import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import os
from glob2 import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageStat
import diffimg

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

train_dir = 'dataset/nomal_data/train'
test_dir = 'dataset/nomal_data/test'
an_dir = 'data1/anomal_data/'

image_size = (48, 48)
batch_size = 64
datagen=ImageDataGenerator(rescale=1./ 255)
train_gen = datagen.flow_from_directory(train_dir,
                                        target_size=image_size,
                                        batch_size=batch_size,
                                        color_mode='grayscale',
                                        class_mode='input',
                                        shuffle=True)

test_gen = datagen.flow_from_directory(test_dir,
                                       target_size=image_size,
                                       batch_size=batch_size,
                                       color_mode='grayscale',
                                       class_mode='input')
an_gen = datagen.flow_from_directory(an_dir,
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     color_mode='grayscale',
                                     class_mode='input')

class convolutional(Model):
    def __init__(self):
        super(convolutional, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
           layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
           layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
           layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
           layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
           layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder3 = convolutional()
autoencoder3.compile(optimizer='adam', loss='mae')
autoencoder3.build((64,48,48,1))
autoencoder3.summary()

history=autoencoder3.fit(train_gen,
                epochs=200,
#                 batch_size=64,
                shuffle=True,
                validation_data=train_gen,
                )

plot1 = plt.figure(1)
plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title('Crossentropy')
plt.ylabel('Crossentropy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")

plot1 = plt.figure(1)
plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title('Crossentropy')
plt.ylabel('Crossentropy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")


plt.show()

plt.show()

# autoencoder3.save('./model/cnn')
tf.saved_model.save(autoencoder3, "./model/cnn")