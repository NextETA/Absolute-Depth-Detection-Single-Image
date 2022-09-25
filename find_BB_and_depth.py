
import numpy as np
import cv2

# takes in a rgbd 4 dimensional image
def find_BB_and_depth(img_rgb, pixel_depths, drawContours=False):
    img = cv2.pyrDown(img_rgb)

    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros([len(contours), 5])

    for i in range(len(contours)):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(contours[i])

        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)