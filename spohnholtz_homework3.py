#Created By: Dan Spohnholtz
#CS4410, Troy University 

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

#Determine Optimal K-Value
X = iris.data
y = iris.target
dataStore = {}

for k in range(1,32,2):
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn, X, y, cv=10)
  print(f'k={k:<2}; mean accuracy={scores.mean():.2%}; ' + f'standard deviation={scores.std():.2%}')

#Use GridSearchCV to Confirm Optimal K-Value
knn2 = KNeighborsClassifier()
grid = {'n_neighbors': np.arange(1,32,2)}
knn_gscv = GridSearchCV(knn2, grid, cv=10)
knn_gscv.fit(X, y)
knn_gscv.best_params_