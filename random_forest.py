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
from IPython import embed

train = np.load('/home/data/preprocessed/shallow/X_train.txt.npy')
y_train = np.load('/home/data/preprocessed/shallow/y_train.txt.npy')
val = np.load('/home/data/preprocessed/shallow/X_val.txt.npy')
y_val = np.load('/home/data/preprocessed/shallow/y_val.txt.npy')
print('DATA LOADED')

pca = PCA(n_components=50)
train = pca.fit_transform(train)
val = pca.transform(val)
print('DATA PREPROCESSING COMPLETED')

print(np.sum(y_train)/np.shape(y_train)[0])
print(np.sum(y_val)/np.shape(y_val)[0])

n_estimators = [1,5,10,100,1000,1500,5000,10000]
max_depth = [1,5,10,20,50,100,1000,None]
for n in n_estimators:
    for d in max_depth:
        clf = RandomForestClassifier(n_estimators=n, criterion='gini', max_depth=d, 
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
            random_state=0, verbose=0, warm_start=False, class_weight=None)

        clf.fit(train,y_train)
        val_predictions = clf.predict(val)
        train_predictions = clf.predict(train)

        val_acc = np.mean(val_predictions == y_val)
        train_acc = np.mean(train_predictions == y_train)
        print('estimators: {}, max depth: {} -> train acc: {}, val acc: {}'.format(n, d, train_acc, val_acc))
embed()