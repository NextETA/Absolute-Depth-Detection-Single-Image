
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