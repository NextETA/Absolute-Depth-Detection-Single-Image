import tensorflow as tf
import numpy as np

def runDNN(Xtrain, Ytrain,Xtest):

    #define the model features
    #FEATURES = ['height/width','depth']
    FEATURES = ['outputs']
    feature_cols = [tf.feature_column.numeric