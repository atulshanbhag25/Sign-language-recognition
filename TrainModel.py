import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d,max_pool_2d
import numpy as np
import cv2
from sklearn.utils import shuffle


#Load Images From C
Images = []
for i in range(0, 1001):
    image = cv2.imread('Dataset/C_images/C_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Images.append(gray_image.reshape(89, 100, 1))


#Load Images from N
for i in range(0, 1001):
    image = cv2.imread('Dataset/N_images/N_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Images.append(gray_image.reshape(89, 100, 1))

    
#Load Images From A
for i in range(0, 1001):
    image = cv2.imread('Dataset/A_images/A_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Images.append(gray_image.reshape(89, 100, 1))

# Creating  OutputVector

outputVectors = []
for i in range(0, 1001):
    outputVectors.append([1, 0, 0])

for i in range(0, 1001):
    outputVectors.append([0, 1, 0])

for i in range(0, 1001):
    outputVectors.append([0, 0, 1])

testImages = []

#Load Images for C
for i in range(0, 1001):
    image = cv2.imread('Dataset/C_test/C_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for N
for i in range(0, 1001):
    image = cv2.imread('Dataset/N_test/N_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for A
for i in range(0, 1001):
    image = cv2.imread('Dataset/A_test/A_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

TestValues = []

for i in range(0, 1001):
    TestValues.append([1, 0, 0])
    
for i in range(0, 1001):
    TestValues.append([0, 1, 0])

for i in range(0, 1001):
    TestValues.append([0, 0, 1])


#  CNN Model
tf.compat.v1.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,3,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)


# Shuffling of training data
Images, outputVectors = shuffle(Images, outputVectors, random_state=0)

# Training model
model.fit(Images, outputVectors, n_epoch=50,
           validation_set = (testImages, TestValues),
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("Model/SignRecog.tfl")







