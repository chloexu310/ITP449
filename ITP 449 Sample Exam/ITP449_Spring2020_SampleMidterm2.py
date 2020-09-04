import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Q1-1
os.chdir('/Users/arpi-admin/Documents/ITP449_Files')
netflix = pd.read_csv('netflix_titles.csv')

# Q1-2
print(netflix.shape)

# Q1-3
print(netflix.info())

# Q1-4
pd.set_option("display.max_columns", None)
print(netflix.head())

# Q1-5
print(netflix.describe())

# Q1-6
print(netflix.isnull().any())

# Q1-7
print(netflix.isnull().sum() / len(netflix) * 100)

# Q1-8
# Drop Director column because over 30% of the rows are missing
# For analysis that requires the following columns with missing values:
# Drop rows where cast, country, date_added and rating columns have missing values. Can't fill since not numerical

# ------------

# Q2-1
moviesCount = netflix.loc[netflix["type"] == "Movie", 'title'].count()
showsCount = netflix.loc[netflix["type"] == "TV Show", 'title'].count()
print(moviesCount / len(netflix) * 100)
print(showsCount / len(netflix) * 100)

# Q2-2
oldMovies = netflix.loc[netflix["type"] == "Movie", ['title', 'release_year']]
oldMovies = oldMovies.sort_values("release_year", ascending=True)
print(oldMovies.iloc[:10])

# Q2-3
newShows = netflix.loc[netflix["type"] == "TV Show", ['title', 'release_year']]
newShows = newShows.sort_values("release_year", ascending=False)
print(newShows.iloc[:10])

# Q2-4
print(netflix['rating'].unique())

# Q2-5
print(netflix.loc[netflix['rating'] == 'R', 'title'].count())

# ----------

# Q3-1
fig = plt.figure()
movies = netflix.loc[(netflix["type"] == "Movie") & (netflix["release_year"] >= 2010)]
plt.plot(movies.groupby("release_year").count()["title"], 'm*--', label='Movie')

shows = netflix.loc[(netflix["type"] == "TV Show") & (netflix["release_year"] >= 2010)]
plt.plot(shows.groupby("release_year").count()["title"], 'g*--', label='TV Show')

plt.title("Netflix Collection Size by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.yticks(np.arange(0, 800, 50))
plt.xticks(np.arange(2010, 2021))
plt.legend()
plt.grid()
plt.show()

# Q3-2
# The collection of movies increased with a peak in 2017
# The collection of TV shows increased with a peak in 2019
# While there was historically more movies than shows,
# the gap closed and there were slightly more shows than movies in 2019
# Overall, the total number of titles decreased as the number of shows grew