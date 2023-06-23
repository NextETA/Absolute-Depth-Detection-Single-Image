
import numpy as np
from linreg_closedform import LinearRegression

# part 0: get the depths

# load all the data
depths = np.load('nyu_dataset_depths.npy')
images = np.load('nyu_dataset_images.npy')
labels = np.load('nyu_dataset_labels.npy')
names = np.load('nyu_dataset_names.npy')