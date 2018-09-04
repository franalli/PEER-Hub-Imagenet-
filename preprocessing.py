import numpy as np 
from IPython import embed

data = np.load('/home/data/X_train.npy')
labels = np.load('/home/data/y_train.npy')

n_rows = data.shape[0]
w = data.shape[1]
d = data.shape[2]
n_validation = int(n_rows * 0.1)
n_train = n_rows - n_validation

X_train = data[:n_train]
y_train = labels[:n_train]
X_val = data[-n_validation:]
y_val = labels[-n_validation:]

x_train_bw, x_val_bw = np.mean(X_train, axis=-1), np.mean(X_val, axis=-1)
x_train_final = np.reshape(x_train_bw, (n_train, w*d))
x_val_final = np.reshape(x_val_bw, (n_validation, w*d))

np.save('/home/data/preprocessed/shallow/X_train.txt', x_train_final)
np.save('/home/data/preprocessed/shallow/y_train.txt', y_train)
np.save('/home/data/preprocessed/shallow/X_val.txt', x_val_final)
np.save('/home/data/preprocessed/shallow/y_val.txt', y_val)

np.save('/home/data/preprocessed/FC/X_train.txt', x_train_final)
np.save('/home/data/preprocessed/FC/y_train.txt', y_train)
np.save('/home/data/preprocessed/FC/X_val.txt', x_val_final)
np.save('/home/data/preprocessed/FC/y_val.txt', y_val)

np.save('/home/data/preprocessed/deep/X_train.txt', X_train)
np.save('/home/data/preprocessed/deep/y_train.txt', y_train)
np.save('/home/data/preprocessed/deep/X_val.txt', X_val)
np.save('/home/data/preprocessed/deep/y_val.txt', y_val)