import tensorflow as tf
import numpy as np

def runDNN(Xtrain, Ytrain,Xtest):

    #define the model features
    #FEATURES = ['height/width','depth']
    FEATURES = ['outputs']
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    #define the DNN regressor
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[1,1])

    #define the input functions for training
    def get_input_fn(xData, yData=N