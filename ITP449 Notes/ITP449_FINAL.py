#Fianl ITP449

housing data

df = pd.read_cvs(xxx)
pd.set_potion(xxx, none)

print(df.info())
print(df["CRIMS"])
#If you see there is a key error, it is following for a key and the key is wrong
#then you look back and you put
print(df["CRIM"])

print(df.CRIM) #This gives you the same ouutput

print(df.iloc[:, "CRIM"]) #Value error happens, what you should do is
print(df.iloc[:0])

print(df.iloc[:,0:3])
print(df.loc[:, ["CRIM"]]) #you can do the name directly (because you know the name but not index



import os
import numpy as np
import pandas as pd
os.chdir("xxx file")

df = pd.read_csv("xx name ")
print(df.info()) #display all the column names
print(df.CRIM) #dot function also gives you the output for "CRIM", you can also do ["CRIM].
print(df[["CRIM","LSTAT"]]) #if you are doing two columns, you need two []!! if not, key error will come out
print(df.describe()) #show the description

crim = df.CRIM
print(crim)
print(df[df["CRIM"]>200]) #give you the value round 200
print(df[df["CRIM"]> 2].loc[:,"CRIM"]) #give you only the series of CRIM column that has a value of 2
print(df)

#ETL and Datawarangling

print(df.isnull().sum())
import matplotlib.pyplot as plt
ply.plot()

