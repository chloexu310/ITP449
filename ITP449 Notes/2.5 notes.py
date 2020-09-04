#print the average sales for Global, North America, European Union and Japan for games with global sales of at least 1 million

import os
import numpy as np
import pandas as pd

os.chdir('/Users/arpi-admin/Documents/ITP449_Files')
pd.set_option('display.max_columns', None)
dfVG = pd.read_csv('vgsales.csv')

dfVG2 = dfVG[dfVG['Gloval_Sales'] >= 1]

print(np.average(dfVG2['Global_Sales']))
print(np.average(dfVG2['NA_Sales']))
print(np.average(dfVG2['EU_Sales']))
print(np.average(dfVG2['JP_Sales']))