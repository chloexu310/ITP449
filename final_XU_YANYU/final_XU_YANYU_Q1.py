#Yanyu Xu
#ITP_449, Spring 2020
#Final Exam
#Question 1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import json
from pandas.io.json import json_normalize

#1. Go to the following link to generate an API key: api.nasa.gov and follow the instructions in the ‘Generate API Key’ section. Save the API key. (1 point)
#1BvbNuhHVUriBUoCUE092WqcqjG9dG7NbNb8GGAW

#2. Using the URL provided above as well as the API Key obtained in step 1, make a get request to the API and store the response. (2 points)
pd.set_option('display.max_columns', None)
url = 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol=1000&api_key=1BvbNuhHVUriBUoCUE092WqcqjG9dG7NbNb8GGAW'
response = requests.get(url)

#3. Display the status code of the response from step 2 and explain what the status code returned means. (1 point)
print(response.status_code)


#4. Store the response with JSON encoding. (1 point)
data = response.json()

#5. Print the JSON encoded data with indent = 5. (1 point)
print(json.dumps(data, indent=5))

#6. Store the data into a DataFrame “marsRover”. What are the dimensions of the DataFrame? What does this mean about the nesting structure of the JSON data? (2 points)
marsRover=json_normalize(data)
print(marsRover.shape)

#the dimensions are id, sol, img_src, earth_date, camera_id, camera_name, camera_rover_id, camera_full_name, rover_id
#rover_name, rover_landing_date, rover_launch_date, rover_status, rover_max_sol, rover_max_date, rover_total_photos, rover_camera


#7. Unnest the ‘photos’ column from marsRover, storing it as a new DataFrame marsRover2. What are the dimensions of marsRover2? (2 points)
marsRover2=json_normalize(data['photos'])
pd.set_option('display.max_columns', None)
print(marsRover2.shape)
#the dimension is id, sol, img_src, earth_date, camera_id, camera_name, camera_rover_id, camera_full_name, rover_id
#rover_name, rover_landing_date, rover_launch_date, rover_status, rover_max_sol, rover_max_date, rover_total_photos, rover_cameras

#8. Is there additional JSON data that is nested within columns of marsRover2? If so, which columns? If so, why didn’t this data unnest in step 7? (2 points)
#YES


#9. Display the number of non-null samples and feature types in marsRover2. (1 point)
marsRover2 = marsRover.dropna()
print(marsRover2.info())

#10. Display the first 5 rows of marsRover2. (1 point)
print(marsRover2.head())

#11. What earth date range is this data for? (1 point)
earthDate = marsRover['earth_date']
print(earthDate.max())
print(earthDate.min())
#This data is 2015-5-30

#12. What are the name(s) of the rover unit(s) that were present? (1 point)
print(marsRover2.rover_name.unique())

#13. What are the name(s) of the camera(s) used? (2 points)
print(marsRover2.camera_name.unique())

#14. How many days elapsed between the rover launch date(s) and landing date(s)? (2 points)
