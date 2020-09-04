# Yanyu Xu
# ITP 449 Spring 2020
# Midterm Exam
# Question 1

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Store the COVID19.csv data into a DataFrame covid.
covid = pd.read_csv('COVID19.csv')

#Print the dimensions of the covid (number of rows and columns).
pd.set_option('display.max_columns', None)
print(covid)

#Print a summary of the features, number of non-null samples and feature types.
print(covid.info())

#Print the first 5 rows, making sure to display all the columns.
pd.set_option("display.max_columns", None)
print(covid.head())

#Print summary statistics (i.e. count, mean, ... 75%, max).
print(covid.describe())

#Print a list of the columns and the percentage of the rows that have missing values for each column.
print(covid.isnull().any())
print(covid.isnull().sum() / len(covid) * 100)