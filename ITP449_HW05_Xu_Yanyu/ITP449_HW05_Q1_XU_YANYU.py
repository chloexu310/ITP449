# Yanyu Xu
# ITP 449 Spring 2020
# HW05
# Question 1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Create a DataFrame to store the home prices data.
home = pd.read_csv('homePrices.csv')
pd.set_option('display.max_columns', None)
df = pd.DataFrame(home)

#Print a summary of the features, number of non-null samples and feature types.
print(home.info())

#Provide an analysis of what needs to be addressed in terms of data wrangling.
dfNoNa = df.dropna(axis=0)
print(dfNoNa.info())
#there is no null data so we do not need to address the data wrangling

#Print summary statistics (i.e. count, mean, ... 75%, max).
print(home.describe())

#Using ‘Number Baths’ and ‘Square Feet’ as features and ‘Price as the target, compute a correlation matrix.
correlation = df[["Number Baths","Square Feet","Price"]]
print(correlation.corr())

#Describe the relationship between all variables.
# there is positive relationship between price and number baths and between price and square feet and number of baths and square feet

#Using ‘Number Baths’ and ‘Square Feet’ as features and ‘Price’ as the target, plot a scatter plot matrix.·
set = df[["Number Baths","Square Feet","Price"]]
sb.pairplot(data=set)
plt.show()

#Plot a scatter plot with ‘Number Baths’ as the independent variable and ‘Price’ as the dependent variable.
plt.scatter(dfNoNa['Number Baths'],dfNoNa['Price'])
plt.xlabel('Number Baths')
plt.ylabel('Price')
plt.show()

#Plot a scatter plot with ‘Square Feet’ as the independent variable and ‘Price’ as the dependent variable.
plt.scatter(dfNoNa['Square Feet'],dfNoNa['Price'])
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()

#Describe further details of the relationship between all variables. Do the relationships meet the requirements for Linear Regression?
# there is a potive relationship between price and square feet and there is also positive relationship between price and number of bath

#Create the Feature Matrix and Target Vector.
X = df[['Number Baths','Square Feet']]
y = df['Price']

print(X.shape)
print(y.shape)

#Split the Feature Matrix and Target Vector into training and testing sets, reserving 30% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#Create and fit a Linear Regression model to the training data.
model = LinearRegression()
model.fit(X_train, y_train)

#Print the intercept and coefficients for the model. Explain what each means.
print(model.intercept_)
print(model.coef_)

#As the number of bath increases, the price increase too
#When square feet increased, the price increased.

#Compute R2 and describe what it means.
print(model.score(X_test,y_test))
#it means the probability to predict correctly

#Create a dummy variable for ‘Type of Home’.
a = pd.get_dummies(df['Type of Home'])
df = pd.concat([df, a], axis=1)
print(df)

#Repeat steps 11-15 to include the dummy variable ‘Detached’ in the Feature Matrix.
#Create the Feature Matrix and Target Vector.
X = df[['Detached','Number Baths','Square Feet']]
y = df['Price']

print(X.shape)
print(y.shape)

#Split the Feature Matrix and Target Vector into training and testing sets, reserving 30% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Create and fit a Linear Regression model to the training data.
model = LinearRegression()
model.fit(X_train, y_train)


#Print the intercept and coefficients for the model. Explain what each means.
print(model.intercept_)
print(model.coef_)

#Compute R2 and describe what it means.
print(model.score(X_test,y_test))

#Which model is better, with or without ‘Type of Home’. Why?
#I think both model is similar because they have similar r2 value around 85%-90%.