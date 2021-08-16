#Troy University, CS4410
#Created By: Daniel Spohnholtz
#Homework 1

import pandas as pd
import matplotlib.pyplot as plt

#Import dimonds.csv from github
url = 'https://raw.githubusercontent.com/dspohnholtz/cs4410/main/diamonds.csv'
df = pd.read_csv(url, index_col=0)

#Display First Seven Rows
print("FIRST SEVEN ROWS OF DATAFRAME")
print(df.loc[0:7])

#Display Last Seven Rows
print("\nLAST SEVEN ROWS OF DATAFRAME")
print(df.tail(7))

#Demostrate DataFrame.describe()
df.describe()

#Demostrate Series.describe()
s = pd.Series(['cut', 'color', 'clarity'])
s.describe()

#Demonstrate Series.unique()
s.unique()

#Demonstrate DataFrame.hist()
df.hist()
plt.tight_layout()