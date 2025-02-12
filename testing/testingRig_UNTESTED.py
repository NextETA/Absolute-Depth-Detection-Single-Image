
import numpy as np
import cv2
import csv
from find_BB_and_depth import find_BB_and_depth
# import load_mat_to_python
from linreg_closedform import LinearRegressionClosedForm as LinearRegression
from PIL import Image
# from NeuralNet import runNeuralNet
import sys


def estimateSize():

    # Part 0: Loading the data with depth from matlab to python

    # convert the matlab file to python
    # load_mat_to_python()

    # Part 1: Loading image and associated depth data into python

    # load all the data
    depths = np.load('data/nyu_dataset_depths.npy')
    images = np.load('data/nyu_dataset_images.npy')
    # labels = np.load('nyu_dataset_labels.npy')
    # names = np.load('nyu_dataset_names.npy')
    # scenes = np.load('nyu_dataset_scenes.npy')

    # Part 2: Import labels n by 4 (img #, bb#, lab_h, lab_w)

    labels = np.loadtxt('data/ImageLabels.dat', delimiter=',')
    n, d = labels.shape

    # array to hold (img#, bb#, lab_h, lab_w, x, y, h, w, d, img_h, img_w)
    imageLabels = np.zeros((n, 11))
    imageLabels[:, 0:4] = labels

    # Part 3: Create bounding boxes for our training images
    for i in range(n):
        print("Testing on: " + str(imageLabels[i]))
        imgNum = int(imageLabels[i, 0])
        imgi = images[:, :, :, imgNum]
        h, w, c = imgi.shape
        bbox = find_BB_and_depth(imgi, depths[:, :, imgNum], False)
        k = int(imageLabels[i, 1])
        imageLabels[i, 4:9] = bbox[k]
        imageLabels[i, 9:11] = (h, w)

    # Part 4: Aggregate training data

    # X train- training height and widths

    train_height = (imageLabels[:, 9] / 2 -
                    imageLabels[:, 6]) * imageLabels[:, 8]
    train_width = (imageLabels[:, 10] / 2 -
                   imageLabels[:, 7]) * imageLabels[:, 8]

    # Y train- training heights and widths
    label_height = imageLabels[:, 2]
    label_width = imageLabels[:, 3]

    # Part 5: Generate the test data
    # image number, bouning box number
    unlabeledWithDescription = np.loadtxt(
        'data/ImageUnLabeled.dat', delimiter=',', usecols=(0, 1))
    n, c = unlabeledWithDescription.shape

    # array to hold (img#, bb#, null, null, x, y, h, w, d, img_h, img_w)
    imageUnLabeled = np.zeros((n, 11))
    imageUnLabeled[:, 0:2] = unlabeledWithDescription

    # Part 6: Fit a Linear Regression with training data
    # do training on linear regression
    linreg_x = LinearRegression(regLambda=1E-8)
    linreg_y = LinearRegression(regLambda=1E-8)

    linreg_x.fit(train_height, label_height)
    linreg_y.fit(train_width, label_width)

    # Part 6: Create bounding boxes for our testing images
    for i in range(n):
        imgNum = int(imageUnLabeled[i, 0])
        imgi = images[:, :, :, imgNum]
        h, w, c = imgi.shape

        # show the image
        # pilimg = Image.fromarray(imgi, 'RGB')
        # pilimg.show()

        # bbox size [k,5] where n is image number, k is num of objects in each image
        # last dimension has x, y, height, width, depth of each bbox in image i
        bbox = find_BB_and_depth(imgi, depths[:, :, imgNum], False)

        # get the bouning box number
        k = int(imageLabels[i, 1])

        # add the bbox values to the imageLabel
        imageLabels[i, 4:9] = bbox[k]

        # add the height width of the image to the imageLabels
        imageLabels[i, 9:11] = (h, w)

    Xtest_height = (imageUnLabeled[:, 10] / 2 -
                    imageUnLabeled[:, 7]) * imageUnLabeled[:, 8]
    Xtest_width = (imageUnLabeled[:, 10] / 2 -
                   imageUnLabeled[:, 7]) * imageUnLabeled[:, 8]

    # Part 6: Fit a Linear Regression with training data
    # do training on linear regression
    linreg_x = LinearRegression(regLambda=1E-8)
    linreg_y = LinearRegression(regLambda=1E-8)

    linreg_x.fit(train_height, label_height)
    linreg_y.fit(train_width, label_width)

    # Part 7: predict heights and widths using linreg
    yHatHeight = linreg_x.predict(Xtest_height)
    yHatWidth = linreg_y.predict(Xtest_width)

    print("yHatHeight:" + str(yHatHeight))
    print("yHatWidth:" + str(yHatWidth))

    y_hat_linreg = np.hstack((yHatHeight, yHatWidth))

    sys.exit()

    # Part 8: Fit a Neural nets with training data

    # re-massage the features
    # width of the bbox, Px, height of bbox, Py, depth
    heightWidth = np.hstack((train_height[:, [0, 2]], train_width[:, [0, 2]]))
    Xtrain = np.c_[heightWidth, train_width[:, 1]]

    # re-massage the ytrain
    #heights and widths in meters
    Ytrain = np.vstack((label_height, label_width)).T

    # get the Xtest data
    heightWidth = np.hstack((Xtest_height[:, [0, 2]], Xtest_width[:, [0, 2]]))
    Xtest = np.c_[heightWidth, train_width[:, 1]]

    y_hat_NN = runNeuralNet(Xtrain, Ytrain, Xtest)


if __name__ == '__main__':
    estimateSize()