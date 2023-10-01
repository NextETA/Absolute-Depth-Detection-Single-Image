import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from BoundingBoxLabeling import BoundingBoxLabeling
from linreg_closedform import LinearRegressionClosedForm as LinearRegression
from PIL import Image

# Load Image and Depth Data
depths = np.load('../data/nyu_dataset_depths.npy')
images = np.load('../data/nyu_dataset_images.npy'