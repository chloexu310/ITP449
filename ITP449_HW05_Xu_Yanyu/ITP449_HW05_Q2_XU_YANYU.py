# Yanyu Xu
# ITP 449 Spring 2020
# HW05
# Question 2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#Create a DataFrame to store the Titanic data.
titanic = pd.read_csv('Titanic.csv')
pd.set_option('display.max_columns', None)
df = pd.DataFrame(titanic)
#Print a summary of the features, number of non-null samples and feature types.
print(titanic.info())

#Provide an analysis of what needs to be addressed in terms of data wrangling.
dfNoNa = df.dropna(axis=0)
print(dfNoNa.info())

#there are null values in age, cabin and embark column.

#Print a list of the columns and the percentage of the rows that have missing values for each column.
print(titanic.isnull().any())
print(titanic.isnull().sum() / len(titanic) * 100)

#Update the DataFrame to account for missing values.
titanic = df.drop(columns=['Cabin'])
titanic = df.dropna(subset=['Age'])

#The target is ‘Survival’. The remainder of the variables are possible features. Drop all columns that are not relevant to predicting ‘Survival’.
dfNoNa1 = df.drop(columns = ['Embarked','PassengerId','Name','Ticket','Fare','SibSp','Parch'])
print(dfNoNa1)

#Create a dummy variables for the remaining categorical variables.
dclass = pd.get_dummies(df['Pclass'])
dsex = pd.get_dummies(df['Sex'])
df = pd.concat([df,dclass], axis=1)
df = pd.concat([df,dsex], axis=1)
df = df.drop(columns=["Sex"])

#Create the Feature Matrix and Target Vector.
X = df[['Pclass','Fare','Survived']]
y = df['Survived']

print(X.shape)
print(y.shape)
#Split the Feature Matrix and Target Vector into training and testing sets, reserving 30% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Create and fit a Logistic Regression model to the training data.
model = LogisticRegression()
model.fit(X_train, y_train)

#Compute the confusion matrix. Describe what each value means.
y_pred = model.predict(X_test)
print(pd.crosstab(y_pred,y_test))

#In this case, which would improve the model more: reducing false positives or false negatives?
#reducing false positives would improve more

#What is the probability of survival for a 25 year old male from 3rd class?
data1 = np.array([3,25.0,0]).reshape(1,3)
predicted = model.predict_proba(data1)
print(predicted)


#What is the probability of survival for a 10 year old female from 1st class?
data2 = np.array([1,10.0,1]).reshape(1,3)
predicted = model.predict_proba(data2)
print(predicted)

