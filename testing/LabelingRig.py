import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from BoundingBoxLabeling import BoundingBoxLabeling
from linreg_closedform import LinearRegressionClosedForm as Linea