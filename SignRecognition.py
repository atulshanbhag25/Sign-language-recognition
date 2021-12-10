import numpy as np
from PIL import Image
import cv2
import imutils
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression


# global variables
bcg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    weight_percent = (basewidth/float(img.size[0]))
    height_size = int((float(img.size[1])*float(weight_percent)))
    img = img.resize((basewidth,height_size), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, weight):
    global bcg
    # initializing the background
    if bcg is None:
        bcg = image.copy().astype("float")
        return

    # update the background based on  weighted average
    cv2.accumulateWeighted(image, bcg, weight)

def segment(image, threshold=20):
    global bcg
    # find  diference between background and  frame
    dif = cv2.absdiff(bcg.astype("uint8"), image)

    # threshold the dif image so that we get the foreground
    img_threshold = cv2.threshold(dif,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the img_threshold image
    (contour, _) = cv2.findContours(img_threshold.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(contour) == 0:
        return
    else:
        #  getting the maximum contour which is the hand area
        img_segment = max(contour, key=cv2.contourArea)
        return (img_threshold, img_segment)

def main():
    # initialize weight 
    weight = 0.6

    # (ROI) coordinates
    top, right, bottom, left = 15, 300, 200, 550


    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # initialize num of frames
    frame_num = 0
    start_recording = False

    # looping
    while(True):
        # get the current frame
        (observed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert ROI to grayscale and blurring

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # look till threshold is reached
        if frame_num < 30:
            run_avg(gray, weight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is img_segment
            if hand is not None:
                # if yes, unpack
                (img_threshold, img_segment) = hand

                # draw the img_segment region and display the frame
                cv2.drawContours(clone, [img_segment + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', img_threshold)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", img_threshold)

        # draw the img_segment hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        frame_num += 1

        # display the frame with img_segment hand
        cv2.imshow("Sign video", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "s", then stop
        if keypress == ord("s"):
            break
        
        if keypress == ord("b"):
            start_recording = True

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(74, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "C"
    elif predictedClass == 1:
        className = "N"
    elif predictedClass == 2:
        className = "A"

    cv2.putText(textImage,"Pedicted Letter : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence percentage : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)




# Model defined
tf.compat.v1.reset_default_graph()
convnet=input_data(shape=[None,74,100,1],name='input')
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

# Load Saved Model
model.load("Model/SignRecog.tfl")

main()
