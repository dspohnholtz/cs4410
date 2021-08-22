from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Display dataset description
diabetes = load_diabetes()
print(diabetes.DESCR)

#Demonstrate shape attribute against dataset
diabetes.data.shape

#Demonstrate shape attribute against target
diabetes.target.shape

#Demonstrate feature_names attribute
diabetes.feature_names

#Explore Data with Pandas
pd.set_option('precision', 4)
pd.set_option('max_columns', 11)
pd.set_option('display.width', None)

diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['DiseaseProg'] = pd.Series(diabetes.target)
diabetes_df.head()

diabetes_df.describe()

#Data Visualization of Features
sample_df = diabetes_df.sample(frac=0.1, random_state=23)

sns.set(font_scale=2)
sns.set_style('whitegrid')

for feature in diabetes.feature_names:
  plt.figure(figsize=(16, 9))
  sns.scatterplot(data=sample_df, x=feature, y='DiseaseProg', hue ='DiseaseProg', palette ='cool', legend=False)

#Splitting the Data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=16)

#Training the Model
linear_regression = LinearRegression()
linear_regression.fit(X=x_train, y=y_train)

for i, name in enumerate(diabetes.feature_names):
  print(f'{name:>10}: {linear_regression.coef_[i]}')

linear_regression.intercept_

#Testing the Model
predicted = linear_regression.predict(x_test)
expected = y_test

predicted[:10]
expected[:10]

#Visualizing Outcomes
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

figure = plt.figure(figsize=(9,9))
axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue = 'Predicted', palette='cool', legend=False)

start=min(expected.min(), predicted.min())
end=max(expected.max(), predicted.max())
axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt.plot([start, end], [start, end], 'k--')

#Regression Model Metrics
metrics.r2_score(expected, predicted)
metrics.mean_squared_error(expected, predicted)

#Choosing the best model
estimators = {
    'LinearRegression': linear_regression,
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'Ridge': Ridge()
}

for estimator_name, estimator_object in estimators.items():
  kfold = KFold(n_splits=10, random_state=11, shuffle=True)
  scores = cross_val_score(estimator=estimator_object, X=diabetes.data, y=diabetes.target, cv=kfold, scoring='r2')
  print(f'{estimator_name:>16}: ' + f'mean of r2 scores={scores.mean():.3f}')