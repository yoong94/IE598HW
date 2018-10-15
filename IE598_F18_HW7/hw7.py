#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:11:00 2018

@author: zhenqinyuan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df_wine.columns = ['Class label', 'Alcohol',
'Malic acid', 'Ash',
'Alcalinity of ash', 'Magnesium',
'Total phenols', 'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity', 'Hue',
'OD280/OD315 of diluted wines',
'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

#rf = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=-1)

rf_parameters = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 200, 300]

for para in rf_parameters:
    rf = RandomForestClassifier(criterion='gini', n_estimators= para, n_jobs=-1)
    #rf.fit(X_train, y_train)
    scores = cross_val_score(estimator=rf, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print ("For %i parameters, the in-sample accuracy is %.4f"%(para, scores.mean())) 
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print ("\t\t the out-of-sample accuracy is %.4f" %accuracy_score(y_test, y_pred))
    print ()
    
rf = RandomForestClassifier(criterion='gini', random_state=1, n_jobs=-1)

feat_labels = df_wine.columns[1:]
params_rf = {'n_estimators': rf_parameters}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf,
scoring='accuracy', cv=10, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_model = grid_rf.best_estimator_

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
         feat_labels[indices[f]],
         importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),
feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

print("My name is Zhenqin Yuan")
print("My NetID is: zyuan10")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")