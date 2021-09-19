#Created By: Dan Spohnholtz
#Troy University, CS4410
#Demo of Elbow Method to Determine Optimal k-value
#for k-Means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
iris = load_iris()

#Convert Data to DataFrame
df=pd.DataFrame(iris['data'])

#Executing k-Means with Range K
wcss = []
K = range(1,10)
for k in K:
  kMeanModel = KMeans(n_clusters=k)
  kMeanModel.fit(df)
  wcss.append(kMeanModel.inertia_)

#Visualizing Data
plt.figure(figsize=(16,8))
plt.plot(K, wcss, 'g-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('The Elbow Method')
plt.show()