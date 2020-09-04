import requests 
import json 
import pandas as pd 
from pandas.io.json import json_normalize 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler  
pd.set_option('display.max_columns', None)  url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=json&select=st_teff,st_mass,st_rad'  
response = requests.get(url) 
print(response.status_code)  
data = response.json()  
print(json.dumps(data, indent=4))  
planets = json_normalize(data)  
print(planets.info())  
planets = planets.dropna() 
print(planets.info())  
print(planets.head())  
scaler = StandardScaler() 
planets_scaled = scaler.fit_transform(planets) 
planets_scaled = pd.DataFrame(planets_scaled, columns=['temp', 'mass', 'rad'])  
print(planets_scaled.head())
