import scipy.io as sio
import numpy as np

print sio.whosmat('modified.mat')

matlab_contents = sio.loadmat('modified.mat')

depths = matlab_contents['dep