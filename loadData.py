
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
    bbox = find_BB_and_depth(imgi)
    # add to the allBBoxes matrix
    np.cat((allBBoxes,bbox),axis=3)

# get the labeled height and widths of k objects in n images
# size = n x k x 2 where last dimension is height and width in meters
labels = [];

# do training on linear regression
model = LinearRegression(regLambda = 0.00001)
model.fit(X, y)

# predict sizes for k objects in n images using linreg

# do training on neural nets

# predict sizes for k objects in n images using neuralnets