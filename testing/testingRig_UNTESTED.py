
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
