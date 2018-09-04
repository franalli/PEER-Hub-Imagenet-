import numpy as np 
import glob
import os
import itertools
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from IPython import embed
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing import image
from keras import optimizers
from IPython import embed

train = np.load('/home/data/preprocessed/deep/X_train.txt.npy')
y_train = np.load('/home/data/preprocessed/deep/y_train.txt.npy')
val = np.load('/home/data/preprocessed/deep/X_val.txt.npy')
y_val = np.load('/home/data/preprocessed/deep/y_val.txt.npy')
print('DATA LOADED')


lr = 0.01

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model = Sequential()
# model.add(Conv2D(32, (7, 7), activation='relu', input_shape=train.shape[1:]))
model.add(Conv2D(32, (7, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

optim = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])


model.fit(train, y_train, epochs=10, verbose=1, batch_size=32)
score = model.evaluate(val, y_val, batch_size=32)

print (DO, lr, score)

