import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#Store the COVID19.csv data into a DataFrame covid.
china = pd.read_csv('chinatemp.csv')
usa = pd.read_csv('ustemp.csv')
globalvirus = pd.read_csv('time_series_19-covid-Confirmed.csv')
merge = pd.read_csv('Data_Merge.csv')

#Print a summary of the features, number of non-null samples and feature types.
print(china.info())
print(usa.info())
print(globalvirus.info())
#How many cities in China does it actually include?

print("How many cities in China does it actually include?", len(china['NAME'].unique()))
print("How many cities in merge does it actually include?", len(merge['provinceEnglishName'].unique()))

#Missing values?	# rows
print("for China")
print(china.isnull().any())
print(china.isnull().sum() / len(china) * 100)

print("for US")
print(usa.isnull().any())
print(usa.isnull().sum() / len(usa) * 100)

print("for Global")
print(globalvirus.isnull().any())
print(globalvirus.isnull().sum() / len(globalvirus) * 100)

#Date range?
maxchina = china.max()
minchina = china.min()
print("max tem in china", maxchina)
print("max tem in china", minchina)

maxusa = usa.max()
minusa = usa.min()
print("max tem in china", maxusa)
print("max tem in china", minusa)

#Any other metadata?

#correlations
#corr, _ = pearsonr(china['TAVG'], usa['TAVG'])