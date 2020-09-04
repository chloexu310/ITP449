#Yanyu Xu
#ITP_449, Spring 2020
#HW03
#Question 3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('vgsales.csv')

#data frame

df.groupby(df['Year']).sum()
df = df.loc[((df['Year'] >= 1980) & (df['Year'] <= 2016)), ['Year', 'Global_Sales', 'NA_Sales']]

#calculate the total global and North American sales by years from 1980 to 2016
value1 =np.sum(df['NA_Sales'])
value2 =np.sum(df['Global_Sales'])
print(value1)
print(value2)


# #Display the line plots for the total sales by years
plt.plot(df.groupby(['Year']).sum().loc[:,'Global_Sales'], 'bo-', label='Global Sales')
plt.plot(df.groupby(['Year']).sum().loc[:,'North American Sales'], 'go--', label='NA_Sales')
plt.title('Video Games Sales by Year')
plt.xlabel('Year')
plt.ylabel('Total Sales(in millions of dollars)')
plt.legend(loc = 'upper left')
plt.grid(which = 'major')
plt.show()