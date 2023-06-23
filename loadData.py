
import numpy as np
from linreg_closedform import LinearRegression

# part 0: get the depths

# load all the data
depths = np.load('nyu_dataset_depths.npy')
images = np.load('nyu_dataset_images.npy')
labels = np.load('nyu_dataset_labels.npy')
names = np.load('nyu_dataset_names.npy')
scenes = np.load('nyu_dataset_scenes.npy')

# get the 200 images we want to train on from images
imgs = [1,2,4,6,16] #add

# get the bounding boxes for the images we want to train on
allBBoxes = []
for i in range (0,len(imgs)):
    imgi = images[i,:,:,:]
    # bbox size [k,5] where n is image number, k is num of objects in each image
    # last dimension has x, y, height, width, depth of each bbox in image i