import pandas as pd
import numpy as np

data = {'ReleaseYear': [2009, 2019, 2015, 2013, 2005],
        'Movie': ["Avatar", "Avengers:Endgame", "Star Wars Ep VII: The Force Awakens"]
        'Expense':[328262000, 475560000, 381704000, 245904000, 208064000]
        'Profit': [1499182307, 1010189087, 843123707, 797747501, 789505652]}



pd.set_option('display.max_columns', None)
df = pd.DataFrame(data)

#multiple column selection
print(df[['Expense', 'Profit']])
print(df.loc[:, [['Expense', 'Profit']]])
print(df.loc[:, ['Expense', 'Profit']])
print(df.iloc[:, [2,3]])
print(df.iloc[:, 2:4])




# row filtering: ReleaseYear >= 2010
print(df['ReleaseYear'] >= 2010)
print(df.ReleaseYear >= 2010)

#Other ones

dfFil = df[df.ReleaseYear >= 2010]
print(dfFil[:, [['Expense', 'Profit']]])
print(dfFil.loc[:, ['Expense', 'Profit']])


#calculate
print(df['Profit'] / 1000000)

#calculate the age of each movie

#calculate expense as a percentage of profit
print(df['Expense']/df['Profit'])
print(df.Expense / df.Profit)

#modifying DataFrames
df['Ones'] = 1
print(df)

dfCheap = df[df.Expense / df.Profit < 0.30]
print(dfCheap.Movie)