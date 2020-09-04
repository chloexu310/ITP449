# Yanyu Xu
# ITP 449 Spring 2020
# Midterm Exam
# Question 2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

covid = pd.read_csv('COVID19.csv')
pd.set_option('display.max_columns', None)
df = pd.DataFrame(covid)

#What are the total number of confirmed cases of coronavirus?
print(np.sum(df['Confirmed']))

#What percentage of cases of coronavirus have resulted in deaths?
deathCount = np.sum(df['Deaths'])
confirmedCount = np.sum(df['Confirmed'])
recoveredCount = np.sum(df['Recovered'])
total = deathCount + confirmedCount + recoveredCount
print(deathCount / total * 100)

#What percentage of cases of coronavirus have resulted in recovery?
print(recoveredCount / total * 100)


#How many countries has the coronavirus spread to?
confirmed = covid[covid['Confirmed'] != 0]
print(len(confirmed['Country'].unique()))


#Which countries have suffered deaths due to the coronavirus?
deathcountry = covid.loc[(covid['Deaths']!=0),['Country']]
print(deathcountry['Country'].unique())


#How many confirmed cases of the coronavirus are there for each country, from greatest to least, as seen below?
case = covid[['Confirmed','Country']]
country = case.groupby(['Country']).sum()
confirm = country.sort_values('Confirmed', ascending = False)
print(confirm)