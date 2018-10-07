#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:43:38 2018

@author: zhenqinyuan
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X = iris.data[:, ]
y = iris.target 




from sklearn.tree import DecisionTreeClassifier

in_sample_accuracy = []
out_of_sample_accuracy = []

random_state = [1,2,3,4,5,6,7,8,9,10]

for i in random_state:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state= i, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    tree = DecisionTreeClassifier(max_depth = 5 , criterion = 'gini', random_state = 1)
    tree.fit(X_train_std, y_train)
    
    y_pred_train = tree.predict(X_train_std)
    y_pred_test = tree.predict(X_test_std)
    insample = accuracy_score(y_train, y_pred_train)
    outsample = accuracy_score(y_test, y_pred_test)
    in_sample_accuracy.append(insample)
    out_of_sample_accuracy.append(outsample)
    
    print('Random State = ', i, '\nin sample accuracy = ', insample, 
         '\nout of sample accuracy = ', outsample, '\n')
    
print('in sample accuracy:')
print("mean: {:.6f}".format(np.mean(in_sample_accuracy)))
print("std: {:.6f}".format(np.std(in_sample_accuracy)),'\n')

print('out of sample accuracy:')
print("mean: {:.6f}".format(np.mean(out_of_sample_accuracy)))
print("std: {:.6f}".format(np.std(out_of_sample_accuracy)),'\n')

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state= 10, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
tree = DecisionTreeClassifier(max_depth = 4 , criterion = 'gini', random_state = 1)
tree.fit(X_train_std, y_train)
cv_scores = cross_val_score(tree, X_train_std, y_train, cv = 10)
print('CV scores = ', cv_scores)
print('mean = ', np.mean(cv_scores))
print('std = ', np.std(cv_scores),'\n')

y_pred = tree.predict(X_test_std)
out_of_sample_accuracy_cv = accuracy_score(y_test, y_pred)
print('out of sample accuracy score is ', out_of_sample_accuracy_cv)

print("My name is Zhenqin Yuan")
print("My NetID is: zyuan10")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


