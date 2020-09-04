import pandas as pd

netflix = pd.read_csv("netflix_titles_clean.csv")
print(netflix.shape)
print(netflix.info())
pd.set_option("display.max_columns",None)
print(netflix.head(5))
print(netflix.describe())
print(netflix.isnull().any())
print(netflix.isnull().sum()/len(netflix))
"""
For rating, date_added, and country, 
it's better to drop the rows with missing values
For director, cast, we can leave them there if they need to be used for analysis or drop the entire columns
because they take up a huge proportion of data.
"""

"""1. tvshow: 0.3158; movie: 0.6842"""
print(netflix.head())
tvShow = netflix[netflix["type"] == "TV Show"]
movie = netflix[netflix["type"] == "Movie"]
tvShowProp = len(tvShow)/len(netflix)
movieProp = len(movie)/len(netflix)
print(tvShowProp)
print(movieProp)

"""2. """
oldest = netflix.sort_values(by=["release_year"],ascending=True)
print(oldest[["release_year","title"]].head(10))
newest = netflix.sort_values(by=["release_year"],ascending=False)
print(newest[["release_year","title"]].head(10))

"""3. """
print(netflix["rating"].unique())

"""4. """
titleR = netflix[netflix["rating"]=="R"]
print(len(titleR))

"""
Q3
"""
import numpy as np
import matplotlib.pyplot as plt
after2009 = netflix[netflix["release_year"]>2009]
# after2009["release_year"] = pd.to_datetime(after2009["release_year"])
# after2009.set_index(netflix["release_year"], inplace=True)
movie = after2009[after2009["type"] == "Movie"]
show = after2009[after2009["type"] == "TV Show"]
movie = movie[["show_id","release_year"]].groupby(["release_year"]).count()
show = show[["show_id","release_year"]].groupby(["release_year"]).count()
plt.plot(movie, color = "m", label="movie", marker="*", linestyle="--")
plt.plot(show, label = "tv show", marker = "*", linestyle = "--", color = "g")
plt.legend(loc="upper right")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.title("Netflix Collection Size by release year")
plt.grid(which="major")
plt.yticks(np.arange(0,751,50))
plt.xticks(np.arange(2010,2021))
plt.show()

"""
before around 2019, netflix's collection contains more movies than tv shows/ 
after 2019, netflix has slightly more tv shows than movies in its collection
the movie size increases until 2017 and then it drops rapidly after
the tv show size increases until 2019 and it sharply drops after that
"""