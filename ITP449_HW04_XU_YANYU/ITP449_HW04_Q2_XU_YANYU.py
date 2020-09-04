#Yanyu Xu
#ITP_449, Spring 2020
#HW04
#Question 2

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('avocado.csv')
pd.plotting.register_matplotlib_converters()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date",inplace=True)

conventional = df[df["Type"]=="conventional"]

#calculate the average avocado price by divide the sum of total revenue by the sum of total volume
data = conventional.groupby("Date").mean()
avgPrice = data["AveragePrice"]
#calculate the total volumes in 100 millions of usd
avgVolume = data["TotalVolume"]/1000000

#ax1 is the first subplot using the common X-axis
ax1 = plt.subplot()
#trend of average price per avocado in red
ax1.plot(avgPrice,color="r")
#color the y sticks
plt.yticks(color='r')
plt.xlabel('Observation Date')
plt.ylabel('Average Price per Avocado',color='r')
#ax2 is the other subplot using the common X-axis
ax2 = ax1.twinx()
#trend of average total volume sold in blue
ax2.plot(avgVolume, color="b")
#color the y sticks
plt.yticks(color='b')
plt.ylabel('Average Total Volume Sold (in 100 millions of $)',color='b')
plt.title('Average Avocado Prices and Volume Sold Over Time')
plt.show()

# What are overall trends in volume of avocados sold over time?
# The overall trends in volume of avocados sold over time is very unstable. The average volume are usually start to
# decease and reach the lowest point from September to December. Then it gradually increase in Janurary and reach to
# the highest average volume around February or March.

# What is the relationship between avocado prices and volume sold?
# The relationship between avocado prices and volume sold is a inverse relationship.





