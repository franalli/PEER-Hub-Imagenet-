import numpy as np 
from IPython import embed
from PIL import Image

data = np.load('/home/data/X_train.npy')
labels = np.load('/home/data/y_train.npy')


n_rows = data.shape[0]
n_train = int(n_rows * 20/100)
n_validation = n_rows - n_train
X_train = data[n_train:]
y_train = labels[n_train:]
X_val = data[:n_validation]
y_val = labels[:n_validation]



embed()