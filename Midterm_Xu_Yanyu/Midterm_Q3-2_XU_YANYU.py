# Yanyu Xu
# ITP 449 Spring 2020
# Midterm Exam
# Question 3 - 2

#Plot the top 10 countries other than China with the most number of confirmed cases of coronavirus. Match formatting to the plot below. Assume ‘Other’ signifies many countries with fewer confirmed cases.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

covid = pd.read_csv('COVID19.csv')
pd.set_option('display.max_columns', None)

case = covid[['Country','Confirmed']].groupby(['Country']).sum()
confirmed = case.sort_values(by=['Confirmed'], ascending=False)
sort = confirmed[1:11]
print(sort)
plt.bar(sort.index, sort.Confirmed, align='center', color='b')
plt.title("Top 10 Countries (Other than China) Affected by Coronavirus")
plt.xlabel("Countries")
plt.ylabel("Number of Confirmed Cases")
plt.grid(which="major")
plt.show()