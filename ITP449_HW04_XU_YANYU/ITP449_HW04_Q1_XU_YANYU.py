#Yanyu Xu
#ITP_449, Spring 2020
#HW04
#Question 1


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('avocado.csv')

df['Date'] = pd.to_datetime(df['Date'])
pd.plotting.register_matplotlib_converters()
#set 'Date' as the index
df.set_index('Date',inplace = True)

#calculate the total revenue
df['Total Revenue'] = df['AveragePrice']*df['TotalVolume']

#slice the dataframe to the conventional type
conventional = df[df['Type']=='conventional']['AveragePrice']

#calculate the average avocado price by conventional
conventionalAvg = conventional.groupby("Date").mean()

#slice the dataframe to the conventional type
organic = df[df['Type']=='organic']['AveragePrice']
#calculate the average avocado price by organic
organicAvg = organic.groupby("Date").mean()


# plot
a = plt.figure()
plt.plot(conventionalAvg, color = "y",linestyle="-",marker = ".",label="conventional")
plt.plot(organicAvg, color = "g",linestyle="-",marker = ".",label="organic")
plt.grid(which = "major")
plt.xlabel("Observation Date")
plt.ylabel("Average Price Per Avocado")
plt.title("Average Avocado Prices Over Time")
plt.legend(loc="upper left")
plt.tick_params(labelsize=6)
a.tight_layout()

plt.show()

# What are the overall trends (for any type of avocado) in avocado prices over time?
# From the data chart, the trend in avocado prices over time is generally increasing since August and reach the highest point somewhere in September.
# Then it decreases fast and reach the lowest point around February to March.

# What are the overall trends in avocado prices for conventional vs. organic?
# The overall trend for conventional and organic are similar. However, the average price for organic
# is usually $0.6 higher than conventional avocado

# What are some anomalies in avocado prices for conventional vs. organic?
# in 2016-03, when organic avocade price is higher than normal, the price for conventional avocade price is sharply lower
# Also, in 2017-03, similar anomaly happened again.


