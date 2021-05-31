# -*- coding: utf-8 -*-
"""
Name: tronghuuphan
Linkined: linkedin.com/in/tronghuuphan/
Handwriting Recognition using MNIST dataset

Written and Trained on Google Colab

Original file is located at
    https://colab.research.google.com/drive/1np9AFFtUIhjnImqcEu9hgRJRTMd5Ewxy
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt

#Load mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Clean data
def clean_data(data, size=(2,26)):
  data = data[:, size[0]:size[1], size[0]:size[1]]
#  data = data.reshape(data.shape[0], (size[1]-size[0])**2)
#  data = data.astype('float32')/255
  data = data.reshape(data.shape[0], (size[1]-size[0]), (size[1]-size[0]), 1 )
  return data

train_images = clean_data(train_images)
test_images = clean_data(test_images)

#Encode the labels
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_labels.shape

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='sigmoid', input_shape=(24,24,1)))
model.add(layers.Conv2D(32, (3,3), activation='sigmoid'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=8, batch_size=128)

#Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test acc:', test_acc)

fig = plt.figure()
numOfEpoch = 8

import numpy as np
plt.plot(np.arange(0, numOfEpoch), history.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), history.history['accuracy'], label='training acc')

plt.title('Acc and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Acc')
plt.legend()

#Predict
index = np.random.randint(0,test_images.shape[0])
plt.imshow(test_images[index].reshape(24,24), cmap='gray')
y_predict = model.predict(test_images[index].reshape(1,24,24,1))
print('Number: ', np.argmax(y_predict))
