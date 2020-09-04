# Yanyu Xu
# ITP 449 Spring 2020
# Midterm Exam
# Question 3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

covid = pd.read_csv('COVID19.csv')
pd.set_option('display.max_columns', None)

#Plot the number of coronavirus related deaths and recoveries over time. Match formatting to the plot below.

covid['Date'] = pd.to_datetime(covid['Date'])

deaths = covid.loc[(covid['Deaths']!=0),['Deaths','Date']]
recovered = covid.loc[(covid['Recovered']!=0),['Recovered','Date']]

x1 = deaths.groupby(['Date']).sum()
x2 = recovered.groupby(['Date']).sum()


plt.plot(x1,'ko-', label='Deaths')
plt.plot(x2, 'go-', label='Recoveries')

pd.to_datetime(covid["Date"]).dt.normalize()

plt.legend(loc="upper left")
plt.grid(which="major")
plt.title("Number of Coronavirus Deaths and Recoveries Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.yticks(np.arange(0, 13000, 2000))
x = ["01/22/2020", "01/25/2020", "01/29/2020","02/01/2020","02/04/2020","02/08/2020","02/11/2020","02/14/2020","02/17/2020"]
x = [dt.datetime.strptime(d,"%m/%d/%Y").date() for d in x]

plt.xticks(x)
plt.show()

