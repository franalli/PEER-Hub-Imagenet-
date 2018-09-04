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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import optimizers
from IPython import embed

train = np.load('/home/data/preprocessed/FC/X_train.txt.npy')
y_train = np.load('/home/data/preprocessed/FC/y_train.txt.npy')
val = np.load('/home/data/preprocessed/FC/X_val.txt.npy')
y_val = np.load('/home/data/preprocessed/FC/y_val.txt.npy')
dims = np.shape(train)[-1]
print('DATA LOADED')

# dims = 1000
# pca = PCA(n_components=dims)
# train = pca.fit_transform(train)
# val = pca.transform(val)
# print('DATA PREPROCESSING COMPLETED')

dropout = [0.1,0.2,0.5,0.8]
learning_rate = [0.001, 0.002, 0.01, 0.02, 0.05, 0.1]

for DO in dropout:
    for lr in learning_rate:
        model = Sequential()
        model.add(Dense(256, input_dim=dims, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DO))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DO))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DO))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DO))
        model.add(Dense(1, activation='sigmoid'))

        optim = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])

        model.fit(train, y_train, epochs=10, verbose=0, batch_size=32)
        score = model.evaluate(val, y_val, batch_size=32)

        print (DO, lr, score)

