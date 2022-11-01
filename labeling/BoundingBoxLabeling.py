import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

# takes in a rgbd 4 dimensional image
def BoundingBoxLabeling(img_rgb, pixel_depths, drawContours = False, imageNum = -1):
    img = cv2.pyrDown(img_rgb)

    r