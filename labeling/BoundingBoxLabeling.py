import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

# takes in a rgbd 4 dimensional image
def BoundingBoxLabeling(img_rgb, pixel_depths, drawContours = False, imageNum = -1):
    img = cv2.pyrDown(img_rgb)

    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros([len(contours), 5])
    listOfBBs = []

    for i in range(len(contours)):
        # get the bounding rect