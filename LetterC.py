import cv2
import numpy as np
import imutils


bcg = None


def avg(image, weight):
    global bcg
    if bcg is None:
        bcg = image.copy().astype("float")
        return

    # update background
    cv2.accumulateWeighted(image, bcg, weight)


def segment(image, threshold=20):
    global bcg
    # find diference between background and  frame
    dif = cv2.absdiff(bcg.astype("uint8"), image)

    # find threshold different image
    Img_threshold = cv2.threshold(dif,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # contours of Img_threshold
    (contour, _) = cv2.findContours(Img_threshold.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # if no contours return home
    if len(contour) == 0:
        return
    else:
        #  get the maximum contour
        img_segment = max(contour, key=cv2.contourArea)
        return (Img_threshold, img_segment)


def main():
    # initialize weight
    weight = 0.6

    #coordinates of ROI
    top, right, bottom, left = 15, 300, 200, 550

    #capturing frame through web cam
    camera = cv2.VideoCapture(0)

    # initialize num of frames
    frame_num = 0
    image_num = 0

    start_recording = False

    while(True):
        (observed, frame) = camera.read()
        if (observed == True):

            # resize the frame
            frame = imutils.resize(frame, width=600)

            # flip the frame
            frame = cv2.flip(frame, 1)

            # cloning the frame
            clone = frame.copy()

            # height and width of the frame
            (height, width) = frame.shape[:2]

            #  ROI
            roi = frame[top:bottom, right:left]

            # converting the ROI to grayscale and blurring
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # look till threshold is reached
            if frame_num < 25:
                avg(gray, weight)
                print(frame_num)
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is img_segment
                if hand is not None:
                    # if yes, unpack
                    (Img_threshold, img_segment) = hand

                    # draw the img_segment and display the frame
                    cv2.drawContours(
                        clone, [img_segment + (right, top)], -1, (0, 0, 255))
                    if start_recording:

                        #  directory in which you want to store the images
                        cv2.imwrite("Dataset/C_test/C_" +
                                    str(image_num) + '.png', Img_threshold)
                        image_num += 1
                    cv2.imshow("Thesholded", Img_threshold)

            # draw the img_segment hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            frame_num += 1

            # display the frame with img_segment hand
            cv2.imshow("Sign Video", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "b", then stop looping
            if keypress == ord("b"):
                start_recording = True

            # if the user pressed "s", then stop looping
            if keypress == ord("s") or image_num > 1000:
                break
            

        else:
            print("Error, Please check your camera")
            break


main()

# free up memory
camera.release()
cv2.destroyAllWindows()
