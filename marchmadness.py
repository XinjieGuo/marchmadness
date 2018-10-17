#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:18:13 2018

"""


#%%
import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

my_data = genfromtxt('NCAA.csv', delimiter=',')
new_data = genfromtxt('2018testing.csv', delimiter = ',')
training = my_data
testing = new_data


X = training[:,[95,96,97,98,99]]
y = training[:,1]
#scale to range 0,1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X.shape)

test_X = testing
min_max_scaler = preprocessing.MinMaxScaler()
test_X = min_max_scaler.fit_transform(test_X)

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X, y)
output = clf.predict_proba(test_X)
np.savetxt("finalresult.csv", output, delimiter = ",")


#%%
import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

my_data = genfromtxt('NCAA.csv', delimiter=',')

training = my_data[:800]
testing = my_data[800:]



X = training[:,[95,96,97,98,99]]
y = training[:,1]
#scale to range 0,1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X.shape)
test_X = testing[:,[95,96,97,98,99]]
test_y = testing[:,1]

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X, y)
print(clf.score(test_X,test_y))
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X = training[:,[95,96,97,98,99]]   
y = training[:,1]
test_X = testing[:,[95,96,97,98,99]]
test_y = testing[:,1]

logit = LogisticRegression(C=1.0)

logit.fit(X,y)
print(logit.score(test_X,test_y))
#print(X_new.shape)

