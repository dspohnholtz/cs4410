#Troy University, CS4410
#Created By: Daniel Spohnholtz
#Homework 4

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
import seaborn as sns

#Load the Dataset
cancer = load_breast_cancer()
print(cancer.DESCR)

#Check Sample and Target Sizes
print(cancer.data.shape)
print(cancer.target.shape)

#Splitting the Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=16)
print(X_train.shape) 
print(X_test.shape)

#Creating the Model
GausNB = GaussianNB()

#Training the Model
GausNB.fit(X_train, y_train)

#Predictors
y_pred = GausNB.predict(X=X_test)

#Estimator Method Score
print(f'{GausNB.score(X_test, y_test):2%}')

#Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

#Classification Report
names = [str(cancer) for cancer in cancer.target_names]
print(classification_report(y_test, y_pred, target_names=names))

#Visualizing the Confusion Matrix
confusion_df = pd.DataFrame(confusion, index=range(2), columns=range(2))
axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')

#K-Fold Cross Validation
kfold = KFold(n_splits=10, random_state=11, shuffle=True)
scores = cross_val_score(estimator=GausNB, X=cancer.data, y=cancer.target, cv=kfold)
print(f'Mean accuracy: {scores.mean():.2%}')
print(f'Accuracy standard deviation: {scores.std():.2%}')

#Running Multiple Models to Find the Best One
estimators = {
    'GaussianNB': GausNB,
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='ibfgs', multi_class='ovr', max_iter=10000),
    'SVC': SVC(gamma='scale')}

for estimator_name, estimator_object, in estimators.items():
  print(f'{estimator_name:>20}: ' + f'mean accuracy={scores.mean():.2%}; ' + f'standard deviation={scores.std():.2%}')
