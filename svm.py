import os  #實驗組
import numpy as np
import json
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
train_X = np.load('train_X.npy') 
train_Y = np.load('train_Y.npy')
c=input("C=");
g=input("gamma=");
scaler = StandardScaler()
parameters = {'clf__C': [0.01, 0.05,0.25, 5,10,100], 'clf__gamma': [0.025,0.04,0.1, 1,10]}
clf = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=50)), ('clf', SVC(probability=True))])
gs = GridSearchCV(estimator=clf, 
                  param_grid=parameters, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(train_X, train_Y)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
clf.fit(train_X, train_Y)
f = open('model.pkl', "wb")
pickle.dump(clf, f)
