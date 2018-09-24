#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:45:16 2018

@author: zhenqinyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
'NOX', 'RM', 'AGE', 'DIS', 'RAD',
'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print (df.info())
print (df.head())
print (df.describe())


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
cbar=True,
annot=True,
square=True,
fmt='.2f',
annot_kws={'size': 15},
yticklabels=cols,
xticklabels=cols)
plt.show()

X = X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)




print( 'Linear Regression:')
#linear
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('Coefficient: %.3f' % slr.coef_[0])

print('Intercept: %.3f' % slr.intercept_)



#mean squared error


print('MSE train: %.3f, test: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

#R^2 score
print('R^2 train: %.3f, test: %.3f' %
(r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))
#residual plot
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print ('\n')

#Ridge
print('Ridge:')
ridge = Ridge()

alpha_space = [0.05, 0.1, 0.5, 1, 1.5, 2, 0.001]

for alpha in alpha_space:
    print ('alpha = ',alpha )
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    print ('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),
           r2_score(y_test, y_test_pred)))
    print('Coefficient: %.3f' % ridge.coef_[0])

    print('Intercept: %.3f' % ridge.intercept_)
    print ('\n')

plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
        


print ('\n')
#lasso
print('Lasso:')
lasso = Lasso()

alpha_space = [0.01, 0.05, 0.1, 0.5,0.8, 1, 2, 0.8]

for alpha in alpha_space:
    print ('alpha = ',alpha )
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print ('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),
           r2_score(y_test, y_test_pred)))
    print('Coefficient: %.3f' % lasso.coef_[0])

    print('Intercept: %.3f' % lasso.intercept_)
    print ('\n')

plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()


print ('\n')
#Elastic Net 

print('Elastic Net :')
elanet = ElasticNet(alpha=1.0)

l1_ratio_space = [0.1,0.5,1,2,2.5]

for l1_ratio in l1_ratio_space:
    print ('l1_ratio = ',l1_ratio )
    elanet.l1_ratio = l1_ratio
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    print ('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' %
          (r2_score(y_train, y_train_pred),
           r2_score(y_test, y_test_pred)))
    print('Coefficient: %.3f' % elanet.coef_[0])

    print('Intercept: %.3f' % elanet.intercept_)
    print ('\n')

plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print ('\n')
print("My name is Zhenqin Yuan")
print("My NetID is: zyuan10")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
