#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:35:55 2018

@author: zhenqinyuan
"""
from sklearn.model_selection import train_test_split

import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',
header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
stratify=y, random_state=42)

# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


#sns.pairplot(df_wine, size=2.5)
#plt.tight_layout()
#plt.show()

import numpy as np
from sklearn.metrics import accuracy_score

cols = df_wine.columns
cm = np.corrcoef(df_wine[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
cbar=True,
annot=False,
square=True,
fmt='.2f',
annot_kws={'size': 15},
yticklabels=cols,
xticklabels=cols)
plt.show()



print("Logestic Regression: ", "\n")
        
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA

lr = LogisticRegression()
lr.fit(X_train, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, lr.predict(X_train)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, lr.predict(X_test)))
print("\n")

print("SVM: ", "\n")

svm = SVC()
svm.fit(X_train_std, y_train)
print("Accuracy score for training samples = ", accuracy_score(y_train, svm.predict(X_train_std)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, svm.predict(X_test_std)))
print("\n")

print("Logestic Regression(PCA): ", "\n")

pca = PCA(n_components=2)
lr_pca = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr_pca.fit(X_train_pca, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, lr_pca.predict(X_train_pca)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, lr_pca.predict(X_test_pca)))
print("\n")

print("SVM(PCA): ", "\n")
svm_pca = SVC()
svm_pca.fit(X_train_pca, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, svm_pca.predict(X_train_pca)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, svm_pca.predict(X_test_pca)))
print("\n")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

print("Logestic Regression(lda): ", "\n")

lr_lda = LogisticRegression()
lr_lda.fit(X_train_lda, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, lr_lda.predict(X_train_lda)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, lr_lda.predict(X_test_lda)))
print("\n")

print("SVM(LDA): ", "\n")
svm_lda = SVC()
svm_lda.fit(X_train_lda, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, svm_lda.predict(X_train_lda)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, svm_lda.predict(X_test_lda)))
print("\n")


from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kcpa = kpca.fit_transform(X_train_std)
X_test_kcpa = kpca.transform(X_test_std)


print("Logestic Regression(kcpa): ", "\n")

lr_kcpa = LogisticRegression()
lr_kcpa.fit(X_train_kcpa, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, lr_kcpa.predict(X_train_kcpa)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, lr_kcpa.predict(X_test_kcpa)))
print("\n")

print("SVM(LDA): ", "\n")
svm_kcpa = SVC()
svm_kcpa.fit(X_train_kcpa, y_train)

print("Accuracy score for training samples = ", accuracy_score(y_train, svm_kcpa.predict(X_train_kcpa)))
print("Accuracy score for testing samples = ", accuracy_score(y_test, svm_kcpa.predict(X_test_kcpa)))
print("\n")

print("My name is Zhenqin Yuan")
print("My NetID is: zyuan10")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

