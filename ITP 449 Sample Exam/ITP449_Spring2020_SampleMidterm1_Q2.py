import pandas as pd

# import dataset into dataframe
frame = pd.read_csv('mtcars.csv')

# print the dataframe
pd.set_option('display.max_columns', None)
print(frame)

# set dataframe index to Car Name column
frame.set_index('Car Name', inplace=True)
print(frame.head())

# create a scatterplot for horsepower vs. miles per gallon
import matplotlib.pyplot as plt

plt.scatter(frame['hp'], frame.mpg, c='m')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon')
#plt.show()

# Create a Linear Regression model for hp and mpg
import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()

y = frame['mpg']
X = frame['hp']
X = X[:, np.newaxis]

# fit the model
model.fit(X, y)
print(model.intercept_)
print(model.coef_)

# display regression line on scatterplot
plt.plot(X, model.predict(X), c='k')
plt.show()
