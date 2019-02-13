---
layout: default
title:  "Predicting IMDB ratings for movies"
permalink: "/predict_movie_ratings/"
---

# <center>Predicting IMDB ratings for movies</center>
## <center>Python</center>

I consider myself an enthusiastic, if not particularly knowledgable, cinephile. When debating between multiple movies to watch, my partner and I always do the "IMDB test" - the movie we choose is the one with the highest average rating on IMDB (I plan  to create a movie recommender system to replace this method at some point, but that's another blog post). I would like to see how easy it could be to predict the average rating for a movie, and what predictors have the most effect.

The dataset I used is the popular [Movies dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) found on Kaggle. The aim is to try different models and find the one with the lowest root mean squared error (RMSE). 

Once I have identified the best performing model, I will create a Flask app to allow users to create their own movies using the given feature options (cast, crew etc) and predict the rating and the revenue it would generate. It would also be worth hooking this up to the OMBd API to get the most up-to-date movie information.


```python
import pandas as pd
import numpy as np
import pickle

from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

from sklearn import linear_model as lm, metrics, tree, ensemble, model_selection as ms, feature_selection, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

%matplotlib inline

pd.options.mode.chained_assignment = None 

np.random.seed(42)

import warnings
warnings.filterwarnings("ignore")
```

```python
sns.set(rc={
    'figure.figsize': (12, 8),
    'font.size': 14
})

# Set palette
sns.set_palette("husl")
```


```python
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
```

## Import and prepare the datasets I'll be using

The Movies dataset has multiple csvs which contain data in json format. I will extract each element from each csv that I want to use and put it into a tabular format. I will then combine all these different tables into one master table.

### credits.csv


```python
credits = pd.read_csv("/Users/jasminepengelly/Desktop/projects/predicting_movie/movie_revenue_predictor/credits.csv")
```


```python
credits.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cast</th>
      <th>crew</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>8844</td>
    </tr>
  </tbody>
</table>
</div>



There is a lot of nesting, so I'll use a for loop to separate out the relevant elements.


```python
all_casts = []
all_crews = []
for i in range(credits.shape[0]):
    cast = eval(credits['cast'][i])
    for x in cast:
        x['movie_id'] = credits['id'][i]
    crew = eval(credits['crew'][i])
    for x in crew:
        x['movie_id'] = credits['id'][i]
    all_casts.extend(cast)
    all_crews.extend(crew)

all_casts = pd.DataFrame(all_casts)
all_crews = pd.DataFrame(all_crews)
```


```python
all_casts.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cast_id</th>
      <th>character</th>
      <th>credit_id</th>
      <th>gender</th>
      <th>id</th>
      <th>movie_id</th>
      <th>name</th>
      <th>order</th>
      <th>profile_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>Woody (voice)</td>
      <td>52fe4284c3a36847f8024f95</td>
      <td>2</td>
      <td>31</td>
      <td>862</td>
      <td>Tom Hanks</td>
      <td>0</td>
      <td>/pQFoyx7rp09CJTAb932F2g8Nlho.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>Buzz Lightyear (voice)</td>
      <td>52fe4284c3a36847f8024f99</td>
      <td>2</td>
      <td>12898</td>
      <td>862</td>
      <td>Tim Allen</td>
      <td>1</td>
      <td>/uX2xVf6pMmPepxnvFWyBtjexzgY.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>Mr. Potato Head (voice)</td>
      <td>52fe4284c3a36847f8024f9d</td>
      <td>2</td>
      <td>7167</td>
      <td>862</td>
      <td>Don Rickles</td>
      <td>2</td>
      <td>/h5BcaDMPRVLHLDzbQavec4xfSdt.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>Slinky Dog (voice)</td>
      <td>52fe4284c3a36847f8024fa1</td>
      <td>2</td>
      <td>12899</td>
      <td>862</td>
      <td>Jim Varney</td>
      <td>3</td>
      <td>/eIo2jVVXYgjDtaHoF19Ll9vtW7h.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>Rex (voice)</td>
      <td>52fe4284c3a36847f8024fa5</td>
      <td>2</td>
      <td>12900</td>
      <td>862</td>
      <td>Wallace Shawn</td>
      <td>4</td>
      <td>/oGE6JqPP2xH4tNORKNqxbNPYi7u.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_crews.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit_id</th>
      <th>department</th>
      <th>gender</th>
      <th>id</th>
      <th>job</th>
      <th>movie_id</th>
      <th>name</th>
      <th>profile_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52fe4284c3a36847f8024f49</td>
      <td>Directing</td>
      <td>2</td>
      <td>7879</td>
      <td>Director</td>
      <td>862</td>
      <td>John Lasseter</td>
      <td>/7EdqiNbr4FRjIhKHyPPdFfEEEFG.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52fe4284c3a36847f8024f4f</td>
      <td>Writing</td>
      <td>2</td>
      <td>12891</td>
      <td>Screenplay</td>
      <td>862</td>
      <td>Joss Whedon</td>
      <td>/dTiVsuaTVTeGmvkhcyJvKp2A5kr.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52fe4284c3a36847f8024f55</td>
      <td>Writing</td>
      <td>2</td>
      <td>7</td>
      <td>Screenplay</td>
      <td>862</td>
      <td>Andrew Stanton</td>
      <td>/pvQWsu0qc8JFQhMVJkTHuexUAa1.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52fe4284c3a36847f8024f5b</td>
      <td>Writing</td>
      <td>2</td>
      <td>12892</td>
      <td>Screenplay</td>
      <td>862</td>
      <td>Joel Cohen</td>
      <td>/dAubAiZcvKFbboWlj7oXOkZnTSu.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52fe4284c3a36847f8024f61</td>
      <td>Writing</td>
      <td>0</td>
      <td>12893</td>
      <td>Screenplay</td>
      <td>862</td>
      <td>Alec Sokolow</td>
      <td>/v79vlRYi94BZUQnkkyznbGUZLjT.jpg</td>
    </tr>
  </tbody>
</table>
</div>



#### Cast


```python
cast = all_casts[['name', 'order', 'gender', 'movie_id']]
cast.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>order</th>
      <th>gender</th>
      <th>movie_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tom Hanks</td>
      <td>0</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tim Allen</td>
      <td>1</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Don Rickles</td>
      <td>2</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jim Varney</td>
      <td>3</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wallace Shawn</td>
      <td>4</td>
      <td>2</td>
      <td>862</td>
    </tr>
  </tbody>
</table>
</div>



#### Crew


```python
crew = all_crews[['name', 'job', 'gender', 'movie_id']]
crew.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>job</th>
      <th>gender</th>
      <th>movie_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John Lasseter</td>
      <td>Director</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joss Whedon</td>
      <td>Screenplay</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Andrew Stanton</td>
      <td>Screenplay</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joel Cohen</td>
      <td>Screenplay</td>
      <td>2</td>
      <td>862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alec Sokolow</td>
      <td>Screenplay</td>
      <td>0</td>
      <td>862</td>
    </tr>
  </tbody>
</table>
</div>



This worked well - I know how the features I would like to use for my models in an easy-to-use format.

### movies_metadata.csv
There are a lot of data in this csv that I do not consider relevant to use for my features. I will have to extract the data I want to use.


```python
movies_metadata = pd.read_csv("/Users/jasminepengelly/Desktop/projects/predicting_movie/movie_revenue_predictor/movies_metadata.csv", low_memory = False)
```


```python
movies_metadata.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>poster_path</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.946943</td>
      <td>/rhIRbceoE9lR4veEXuwCC2wARtG.jpg</td>
      <td>[{'name': 'Pixar Animation Studios', 'id': 3}]</td>
      <td>[{'iso_3166_1': 'US', 'name': 'United States o...</td>
      <td>1995-10-30</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>17.015539</td>
      <td>/vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg</td>
      <td>[{'name': 'TriStar Pictures', 'id': 559}, {'na...</td>
      <td>[{'iso_3166_1': 'US', 'name': 'United States o...</td>
      <td>1995-12-15</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
    </tr>
  </tbody>
</table>
</div>



Again, there will be a lot of unpicking required here. I'll intuitively select the features I would like to use and clean them, for example dummifying the __belongs_to_collection__ column.


```python
metadata = movies_metadata[['title', 'id', 'budget',  'revenue', 'runtime', 'vote_average', 
                            'vote_count', 'belongs_to_collection']].copy()
```


```python
metadata['belongs_to_collection'].fillna('0', inplace = True)
```


```python
collection = metadata[['belongs_to_collection']].copy()
collection[collection != '0'] = '1'
metadata['belongs_to_collection'] = collection
```


```python
metadata.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>belongs_to_collection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>862</td>
      <td>30000000</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>8844</td>
      <td>65000000</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grumpier Old Men</td>
      <td>15602</td>
      <td>0</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Waiting to Exhale</td>
      <td>31357</td>
      <td>16000000</td>
      <td>81452156.0</td>
      <td>127.0</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Father of the Bride Part II</td>
      <td>11862</td>
      <td>0</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



I want to also include the movie genre as a feature, but this is nested. I will extract the relevant column and transform it into a workable format. Since movies can belong to one genre, I will have to take this into account by creating a __tmp__ column which will as an indicator as to whether that movie belongs to that genre - this will come in handy later.


```python
g = movies_metadata[['id', 'genres']]
```


```python
genlist = []

for i in range(g.shape[0]):
    gen = eval(g['genres'][i])
    for each in gen:
        each['id'] = g['id'][i]
    genlist.extend(gen)
genre = pd.DataFrame(genlist)
```


```python
genre = genre.rename(columns={'name':'genre'})
genre = genre[['id', 'genre']]
genre['tmp'] = 1
genre.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>genre</th>
      <th>tmp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>862</td>
      <td>Animation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>862</td>
      <td>Comedy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>862</td>
      <td>Family</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8844</td>
      <td>Adventure</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8844</td>
      <td>Fantasy</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



I will now create a pivot table out of my __genre__ DataFrame, filling all the null cells (ie. the ones  that do not contain 1s) with 0. This means every movies' genre is indicated by a 1 or 0.

I can now merge my __genres__ data frame to my metadata DataFrame.


```python
pivot = genre.pivot_table('tmp', 'id', 'genre', fill_value=0)
flattened = pd.DataFrame(pivot.to_records())
metadata_genre = pd.merge(metadata, flattened, on = 'id', how = 'left')
metadata_genre.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>belongs_to_collection</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Aniplex</th>
      <th>BROSTA TV</th>
      <th>Carousel Productions</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Foreign</th>
      <th>GoHands</th>
      <th>History</th>
      <th>Horror</th>
      <th>Mardock Scramble Production Committee</th>
      <th>Music</th>
      <th>Mystery</th>
      <th>Odyssey Media</th>
      <th>Pulser Productions</th>
      <th>Rogue State</th>
      <th>Romance</th>
      <th>Science Fiction</th>
      <th>Sentai Filmworks</th>
      <th>TV Movie</th>
      <th>Telescene Film Group Productions</th>
      <th>The Cartel</th>
      <th>Thriller</th>
      <th>Vision View Entertainment</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>862</td>
      <td>30000000</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>8844</td>
      <td>65000000</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grumpier Old Men</td>
      <td>15602</td>
      <td>0</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Waiting to Exhale</td>
      <td>31357</td>
      <td>16000000</td>
      <td>81452156.0</td>
      <td>127.0</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Father of the Bride Part II</td>
      <td>11862</td>
      <td>0</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now I'll add the cast to the main dataframe, selecting the lead and the supporting actor as two of the features I would like to use.


```python
lead = cast[cast['order'] == 0]
lead = lead[['movie_id', 'name']]
lead = lead.rename(columns={'movie_id':'id'})
lead = lead.rename(columns={'name':'lead'})
lead.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>lead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>862</td>
      <td>Tom Hanks</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8844</td>
      <td>Robin Williams</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Number of rows before dropping those with null values:",len(metadata_genre))
metadata_genre.dropna(inplace = True)
print("Number of rows after dropping those with null values:",len(metadata_genre))
```

    Number of rows before dropping those with null values: 45466
    Number of rows after dropping those with null values: 42839



```python
metadata_genre['id'] = metadata_genre['id'].astype('int64')
metadata_genre_lead = pd.merge(metadata_genre, lead, on = 'id', how = 'left')
```


```python
supporting = cast[cast['order'] == 1]
supporting = supporting[['movie_id', 'name']]
supporting = supporting.rename(columns={'movie_id':'id'})
supporting = supporting.rename(columns={'name':'supporting'})
metadata_genre_lead_supporting = pd.merge(metadata_genre_lead, supporting, on = 'id', how = 'left')
```

Now I'll add the director.


```python
director = crew[crew['job'] == 'Director']
director = director[['movie_id', 'name']]
director = director.rename(columns={'movie_id':'id', 'name':'director'})
```


```python
dataset = pd.merge(metadata_genre_lead_supporting, director, on = 'id', how = 'left')
print("Number of rows before dropping those with null values:",len(dataset))
dataset.dropna(inplace = True)
print("Number of rows after dropping those with null values:",len(dataset))
```

    Number of rows before dropping those with null values: 47632
    Number of rows after dropping those with null values: 37580


Time for some cleaning. I'll remove all rows with null revenue and budget and where the vote count was below 50. I'll also convert each column to the appropriate data type and delete the duplicates.


```python
final_dataset = dataset.loc[(dataset['budget'] != '0') & (dataset['revenue'] != 0)]
final_dataset['budget'] = final_dataset['budget'].astype('float64')
final_dataset['belongs_to_collection'] = final_dataset['belongs_to_collection'].astype('int64')
final_dataset = final_dataset[final_dataset['vote_count'] > 50]
```


```python
print("Before duplicates dropped:", len(final_dataset))
final_dataset.drop_duplicates(inplace=True)
print("After duplicates dropped:", len(final_dataset))
```

    Before duplicates dropped: 4676
    After duplicates dropped: 4632



```python
final_dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>belongs_to_collection</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Aniplex</th>
      <th>BROSTA TV</th>
      <th>Carousel Productions</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Foreign</th>
      <th>GoHands</th>
      <th>History</th>
      <th>Horror</th>
      <th>Mardock Scramble Production Committee</th>
      <th>Music</th>
      <th>Mystery</th>
      <th>Odyssey Media</th>
      <th>Pulser Productions</th>
      <th>Rogue State</th>
      <th>Romance</th>
      <th>Science Fiction</th>
      <th>Sentai Filmworks</th>
      <th>TV Movie</th>
      <th>Telescene Film Group Productions</th>
      <th>The Cartel</th>
      <th>Thriller</th>
      <th>Vision View Entertainment</th>
      <th>War</th>
      <th>Western</th>
      <th>lead</th>
      <th>supporting</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>862</td>
      <td>30000000.0</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Tom Hanks</td>
      <td>Tim Allen</td>
      <td>John Lasseter</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>8844</td>
      <td>65000000.0</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Robin Williams</td>
      <td>Jonathan Hyde</td>
      <td>Joe Johnston</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Heat</td>
      <td>949</td>
      <td>60000000.0</td>
      <td>187436818.0</td>
      <td>170.0</td>
      <td>7.7</td>
      <td>1886.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Al Pacino</td>
      <td>Robert De Niro</td>
      <td>Michael Mann</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sudden Death</td>
      <td>9091</td>
      <td>35000000.0</td>
      <td>64350171.0</td>
      <td>106.0</td>
      <td>5.5</td>
      <td>174.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Jean-Claude Van Damme</td>
      <td>Powers Boothe</td>
      <td>Peter Hyams</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GoldenEye</td>
      <td>710</td>
      <td>58000000.0</td>
      <td>352194034.0</td>
      <td>130.0</td>
      <td>6.6</td>
      <td>1194.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Pierce Brosnan</td>
      <td>Sean Bean</td>
      <td>Martin Campbell</td>
    </tr>
  </tbody>
</table>
</div>



#### Dealing with multiple directors
Though I managed to drop the duplicates above, I can see that some films have multiple directors. This leaves duplicated films that weren't removed with the above because the __director__ columns are different. 

My intention was originally to dummify the directors and merge them with the original data frame. However, the data set became too large to process on my laptop so I'll try again but with the least frequently appearing directors removed to make the dataset more manageable.


```python
final_dataset['dir_count'] = final_dataset.groupby('director')['director'].transform('count')
```


```python
final_dataset['dir_count'] = final_dataset.groupby('director')['director'].transform('count')
final_dataset = final_dataset[final_dataset["dir_count"] >= 5]
final_dataset_wo_dir = final_dataset.drop("director", axis=1).drop_duplicates(keep="first")
directors = final_dataset[["id", "director"]]
```

I'll now export the *final_dataset_wo_dir* for use in my next analysis, which will require the same cleaned features.


```python
final_dataset_wo_dir.to_csv("movies_wo_dir.csv")
```

## Exploratory data analysis
Now it's time to look and at all of my features and look out for things like multicollinearity that could affect my model.

First, I'll define my predictors (without __director__) and response variables.


```python
X = ['budget', 'runtime', 'vote_count', 'belongs_to_collection', 'Action', 'Adventure', 
              'Animation', 'Aniplex', 'BROSTA TV', 'Carousel Productions', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Foreign', 'GoHands', 'History', 'Horror', 'Mardock Scramble Production Committee',
              'Music', 'Mystery', 'Odyssey Media', 'Pulser Productions', 'Rogue State', 'Romance', 'Science Fiction',
              'Sentai Filmworks', 'TV Movie', 'Telescene Film Group Productions', 'The Cartel', 'Thriller', 
              'Vision View Entertainment', 'War', 'Western', 'lead', 'supporting', 'revenue']

y = 'vote_average'
```

Now I'll check for correlation between the predictors.


```python
sns.heatmap(final_dataset_wo_dir.drop(['title', 'id'], axis=1).corr(), vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(10, 220, sep=80, n=7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a3a7dbeb8>




![Correlation Matrix](https://raw.githubusercontent.com/JazPeng/assets/master/movies/correlation_rating.png)


__vote\_average__, my response variable, isn't highly correlated with any other variable, so there are not currently any stand-out features that I think will be particularly predictive.

Within my features, there are four stronger correlations :

* __revenue__ and __budget__
* __vote\_count__ and __budget__
* __revenue__ and __vote\_count__
* __family__ and __animation__

I'll look at these relationships in greater detail below.

#### *revenue* and *budget*


```python
sns.regplot(x='revenue', y='budget', data=final_dataset_wo_dir)
plt.title('Revenue vs Budget')
plt.xlabel('Revenue')
plt.ylabel('Budget')
```

![Revenue vs Budget](https://raw.githubusercontent.com/JazPeng/assets/master/movies/rev_vs_budget.png)


Since the points are very clustered together, I will visualise this a double-log scale to see if I can get a better view of the relationship.


```python
rev_vs_bug = sns.regplot(x='revenue', y='budget', data=final_dataset_wo_dir)
plt.title('Revenue vs Budget')
plt.xlabel('Revenue')
plt.ylabel('Budget')
rev_vs_bug.set(xscale="log", yscale="log")
```

![Revenue vs Budget double log scale](https://raw.githubusercontent.com/JazPeng/assets/master/movies/rev_vs_budget_log.png)


The relationship is strong at 0.69, although it is not represented perfectly by a straight line. The correlation coefficient between __revenue__ and __budget__. It also makes intuitive sense that movies that have larger budgets will attract larger audiences.

#### *revenue* and *vote_count*


```python
sns.regplot(x='revenue', y='vote_count', data=final_dataset_wo_dir)
plt.title('Revenue vs Vote')
plt.xlabel('Revenue')
plt.ylabel('Vote')
```

![Revenue vs Vote Count](https://raw.githubusercontent.com/JazPeng/assets/master/movies/rev_vs_vcount.png)



```python
rev_vs_vote = sns.regplot(x='revenue', y='vote_count', data=final_dataset_wo_dir)
plt.title('Revenue vs Vote')
plt.xlabel('Revenue')
plt.ylabel('Vote')
rev_vs_vote.set(xscale="log", yscale="log")
```

![Revenue vs Vote Count double log](rev_vs_vcount_log.png)


Again, the relationship is better illustrated on a double-log scale and not perfectly represented by a straight line. The correlation coefficient between __revenue__ and __vote_count__ is even more significant at 0.74. As with the relationship between __revenue__ and __budget__, the larger an audience the higher the number of votes can be expected.

#### *vote\_count* and *budget*


```python
sns.regplot(x='budget', y='vote_count', data=final_dataset_wo_dir)
plt.title('Vote Count vs Budget')
plt.xlabel('Budget')
plt.ylabel('Vote Count')
```

![Vote Count vs Budget](vcount_budget.png)

```python
votecount_vs_budget = sns.regplot(x='budget', y='vote_count', data=final_dataset_wo_dir)
plt.title('Vote Count vs Budget')
plt.xlabel('Budget')
plt.ylabel('Vote Count')
votecount_vs_budget.set(xscale="log", yscale="log")
```

![Vote Count vs Budget](https://raw.githubusercontent.com/JazPeng/assets/master/movies/vcount_budget_log.png)


Although there is a correlation coefficient of 0.52, the relationship is again not exactly linear. It appears films with a larger budget spent producing them tend to generate more votes, which makes sense.

#### *Family* and *Animation*


```python
# Both
both = len(final_dataset_wo_dir[(final_dataset_wo_dir['Family'] == 1) & (final_dataset_wo_dir['Animation'] == 1)])

# Animation
ani = len(final_dataset_wo_dir[final_dataset_wo_dir['Animation'] == 1])

# Family
fam = len(final_dataset_wo_dir[final_dataset_wo_dir['Family'] == 1])
```


```python
venn2(subsets = (fam-both, ani-both, 67), 
      set_labels = ('Family', 'Animation'))
plt.show()

```

![Venn Diagram](https://raw.githubusercontent.com/JazPeng/assets/master/movies/venn.png)


There are, in total, 78 films of genre 'Animation' and 186  films of genre 'Family'. While most __Animation__ films are also classified as __Family__, the same cannot be said for the other way around. Since this is the case, I will keep both predictors as they are in my models.

### The highest and lowest rated films
#### Highest rated


```python
final_dataset_wo_dir[['title', 'vote_average']].sort_values(by = 'vote_average', ascending = False).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>849</th>
      <td>The Godfather</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>13316</th>
      <td>The Dark Knight</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>537</th>
      <td>Schindler's List</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>5766</th>
      <td>Spirited Away</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>2985</th>
      <td>Fight Club</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>300</th>
      <td>Pulp Fiction</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Psycho</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Leon: The Professional</td>
      <td>8.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Lowest rated


```python
final_dataset_wo_dir[['title', 'vote_average']].sort_values(by = 'vote_average', ascending = True).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13786</th>
      <td>Disaster Movie</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>12287</th>
      <td>Epic Movie</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>6795</th>
      <td>Gigli</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>11468</th>
      <td>Date Movie</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>13179</th>
      <td>Meet the Spartans</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>14420</th>
      <td>Street Fighter: The Legend of Chun-Li</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>19506</th>
      <td>Jack and Jill</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>22997</th>
      <td>The Canyons</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>30203</th>
      <td>The Boy Next Door</td>
      <td>4.1</td>
    </tr>
    <tr>
      <th>1557</th>
      <td>Speed 2: Cruise Control</td>
      <td>4.1</td>
    </tr>
  </tbody>
</table>
</div>



### Dummy variables
As well as making the variables in the __final_dataset_wo_dir__ dummy, I also need to manually add the directors. I'll do this by creating a pivot table with the directors as columns, the movie id as the row and the values as either 0 or 1.

Since the feature importance of dummy variables relies on the baseline of the first variable that is dropped, I'll make a note of whom that is in __lead__, __supporting__ and __director__ for later analysis.


```python
dropped_dummies = final_dataset_wo_dir[['lead', 'supporting']].head(1)
dropped_dummies
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lead</th>
      <th>supporting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tom Hanks</td>
      <td>Tim Allen</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies = pd.get_dummies(final_dataset_wo_dir, columns=['lead', 'supporting'], drop_first=True)
```


```python
dummies.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>belongs_to_collection</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Aniplex</th>
      <th>BROSTA TV</th>
      <th>Carousel Productions</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Foreign</th>
      <th>GoHands</th>
      <th>History</th>
      <th>Horror</th>
      <th>Mardock Scramble Production Committee</th>
      <th>Music</th>
      <th>Mystery</th>
      <th>Odyssey Media</th>
      <th>Pulser Productions</th>
      <th>Rogue State</th>
      <th>Romance</th>
      <th>Science Fiction</th>
      <th>Sentai Filmworks</th>
      <th>TV Movie</th>
      <th>Telescene Film Group Productions</th>
      <th>The Cartel</th>
      <th>Thriller</th>
      <th>Vision View Entertainment</th>
      <th>War</th>
      <th>Western</th>
      <th>dir_count</th>
      <th>lead_Aaron Eckhart</th>
      <th>lead_Aaron Taylor-Johnson</th>
      <th>lead_Adam Brody</th>
      <th>lead_Adam Sandler</th>
      <th>lead_Adrien Brody</th>
      <th>lead_Adrienne Barbeau</th>
      <th>lead_Aileen Quinn</th>
      <th>lead_Al Pacino</th>
      <th>lead_Alan Arkin</th>
      <th>lead_Alec Baldwin</th>
      <th>lead_Alex D. Linz</th>
      <th>lead_Alex Frost</th>
      <th>lead_Alex Pettyfer</th>
      <th>lead_Alexa PenaVega</th>
      <th>lead_Alexander Skarsgård</th>
      <th>lead_Ali Larter</th>
      <th>lead_Alison Lohman</th>
      <th>lead_Alyson Hannigan</th>
      <th>lead_Amanda Bynes</th>
      <th>lead_Amanda Seyfried</th>
      <th>lead_Amber Tamblyn</th>
      <th>lead_Amber Valletta</th>
      <th>lead_Amy Adams</th>
      <th>lead_Amy Schumer</th>
      <th>lead_Andrew Dice Clay</th>
      <th>lead_Andrew Garfield</th>
      <th>lead_Andy García</th>
      <th>lead_Andy Samberg</th>
      <th>lead_Angelina Jolie</th>
      <th>lead_Anika Noni Rose</th>
      <th>lead_Anna Faris</th>
      <th>lead_Anna Kendrick</th>
      <th>lead_Anne Hathaway</th>
      <th>lead_Ansel Elgort</th>
      <th>lead_Anthony Hopkins</th>
      <th>lead_Anthony Perkins</th>
      <th>lead_Anthony Rapp</th>
      <th>lead_Antonio Banderas</th>
      <th>lead_Arnold Schwarzenegger</th>
      <th>lead_Arthur Hill</th>
      <th>lead_Asa Butterfield</th>
      <th>lead_Ashley Judd</th>
      <th>lead_Ashton Kutcher</th>
      <th>lead_Aubrey Peeples</th>
      <th>lead_Audrey Hepburn</th>
      <th>lead_Audrey Tautou</th>
      <th>lead_Auli'i Cravalho</th>
      <th>lead_Barbara Harris</th>
      <th>lead_Barney Clark</th>
      <th>lead_Barret Oliver</th>
      <th>lead_Barrie Ingham</th>
      <th>lead_Ben Affleck</th>
      <th>lead_Ben Kingsley</th>
      <th>lead_Ben Stiller</th>
      <th>lead_Ben Whishaw</th>
      <th>lead_Benedict Cumberbatch</th>
      <th>lead_Benicio del Toro</th>
      <th>lead_Benjamin Bratt</th>
      <th>lead_Benjamin Walker</th>
      <th>lead_Bill Murray</th>
      <th>lead_Bill Nighy</th>
      <th>lead_Bill Pullman</th>
      <th>lead_Billy Bob Thornton</th>
      <th>lead_Billy Campbell</th>
      <th>lead_Billy Crystal</th>
      <th>lead_Björk</th>
      <th>lead_Blake Jenner</th>
      <th>lead_Blake Lively</th>
      <th>lead_Bob Hoskins</th>
      <th>lead_Bob Newhart</th>
      <th>lead_Bobby Campo</th>
      <th>lead_Bobby Driscoll</th>
      <th>lead_Bodil Jørgensen</th>
      <th>lead_Brad Davis</th>
      <th>lead_Brad Pitt</th>
      <th>lead_Bradley Cooper</th>
      <th>lead_Brady Corbet</th>
      <th>lead_Brandon Lee</th>
      <th>lead_Brandon Routh</th>
      <th>lead_Breckin Meyer</th>
      <th>lead_Brendan Fraser</th>
      <th>lead_Brenton Thwaites</th>
      <th>lead_Brian Bedford</th>
      <th>lead_Brian O'Halloran</th>
      <th>lead_Briana Evigan</th>
      <th>lead_Britt Robertson</th>
      <th>lead_Bruce Campbell</th>
      <th>lead_Bruce Dern</th>
      <th>lead_Bruce Reitherman</th>
      <th>lead_Bruce Willis</th>
      <th>lead_Bryan Cranston</th>
      <th>lead_Bryce Dallas Howard</th>
      <th>lead_Burt Lancaster</th>
      <th>lead_Burt Reynolds</th>
      <th>lead_Cameron Diaz</th>
      <th>lead_Camilla Belle</th>
      <th>lead_Carice van Houten</th>
      <th>lead_Carmen Maura</th>
      <th>lead_Carroll Baker</th>
      <th>lead_Cary Elwes</th>
      <th>lead_Cary Grant</th>
      <th>lead_Casper Van Dien</th>
      <th>lead_Cate Blanchett</th>
      <th>lead_Catherine Deneuve</th>
      <th>lead_Catherine Zeta-Jones</th>
      <th>lead_Cecilia Roth</th>
      <th>lead_Chadwick Boseman</th>
      <th>lead_Chang Chen</th>
      <th>lead_Channing Tatum</th>
      <th>lead_Charles Bronson</th>
      <th>lead_Charlie Chaplin</th>
      <th>lead_Charlie Hunnam</th>
      <th>lead_Charlie Sheen</th>
      <th>lead_Charlie Tahan</th>
      <th>lead_Charlize Theron</th>
      <th>lead_Chevy Chase</th>
      <th>lead_Chow Yun-fat</th>
      <th>lead_Chris Evans</th>
      <th>lead_Chris Hemsworth</th>
      <th>lead_Chris O'Donnell</th>
      <th>lead_Chris Pine</th>
      <th>lead_Chris Riggi</th>
      <th>lead_Chris Rock</th>
      <th>lead_Chris Tucker</th>
      <th>lead_Christian Bale</th>
      <th>lead_Christian Slater</th>
      <th>lead_Christina Applegate</th>
      <th>lead_Christina Ricci</th>
      <th>lead_Christine Hargreaves</th>
      <th>lead_Christoph Waltz</th>
      <th>lead_Christopher Guest</th>
      <th>lead_Christopher Lambert</th>
      <th>lead_Christopher Plummer</th>
      <th>lead_Christopher Reeve</th>
      <th>lead_Christopher Walken</th>
      <th>lead_Cillian Murphy</th>
      <th>lead_Claire Danes</th>
      <th>lead_Claire Trevor</th>
      <th>lead_Clark Gable</th>
      <th>lead_Cleavon Little</th>
      <th>lead_Clint Eastwood</th>
      <th>lead_Clive Owen</th>
      <th>lead_Colin Farrell</th>
      <th>lead_Colin Firth</th>
      <th>lead_Craig T. Nelson</th>
      <th>lead_Craig Wasson</th>
      <th>lead_Cuba Gooding Jr.</th>
      <th>lead_Daisy Ridley</th>
      <th>lead_Dakota Blue Richards</th>
      <th>lead_Dakota Johnson</th>
      <th>lead_Dan Aykroyd</th>
      <th>lead_Dane DeHaan</th>
      <th>lead_Daniel Brühl</th>
      <th>lead_Daniel Craig</th>
      <th>lead_Daniel Day-Lewis</th>
      <th>lead_Daniel Radcliffe</th>
      <th>lead_Danny Aiello</th>
      <th>lead_Danny DeVito</th>
      <th>lead_Danny McBride</th>
      <th>lead_Danny Trejo</th>
      <th>lead_Dany Boon</th>
      <th>lead_David Arquette</th>
      <th>lead_David Duchovny</th>
      <th>lead_David Emge</th>
      <th>lead_David Naughton</th>
      <th>lead_David Niven</th>
      <th>lead_David Strathairn</th>
      <th>lead_Debra Winger</th>
      <th>lead_Dee Wallace</th>
      <th>lead_Demi Moore</th>
      <th>lead_Demián Bichir</th>
      <th>lead_Dennis Quaid</th>
      <th>lead_Denzel Washington</th>
      <th>lead_Derek Jacobi</th>
      <th>lead_Derek Luke</th>
      <th>lead_Dev Patel</th>
      <th>lead_Diana Ross</th>
      <th>lead_Dominic Cooper</th>
      <th>lead_Dominique Pinon</th>
      <th>lead_Donald Pleasence</th>
      <th>lead_Donald Sutherland</th>
      <th>lead_Drew Barrymore</th>
      <th>lead_Duane Jones</th>
      <th>lead_Dustin Hoffman</th>
      <th>lead_Dwayne Johnson</th>
      <th>lead_Ed Harris</th>
      <th>lead_Eddie Griffin</th>
      <th>lead_Eddie Murphy</th>
      <th>lead_Eddie Redmayne</th>
      <th>lead_Edmund Gwenn</th>
      <th>lead_Eduardo Noriega</th>
      <th>lead_Edward Norton</th>
      <th>lead_Eli Marienthal</th>
      <th>lead_Elijah Wood</th>
      <th>lead_Elisha Cuthbert</th>
      <th>lead_Elizabeth Berkley</th>
      <th>lead_Elizabeth Hurley</th>
      <th>lead_Elizabeth Taylor</th>
      <th>lead_Ellar Coltrane</th>
      <th>lead_Elle Fanning</th>
      <th>lead_Ellen Burstyn</th>
      <th>lead_Ellen Page</th>
      <th>lead_Emile Hirsch</th>
      <th>lead_Emilio Estevez</th>
      <th>lead_Emily Barclay</th>
      <th>lead_Emily Blunt</th>
      <th>lead_Emily Browning</th>
      <th>lead_Emily Watson</th>
      <th>lead_Emma Thompson</th>
      <th>...</th>
      <th>supporting_Peter Sellers</th>
      <th>supporting_Phil Harris</th>
      <th>supporting_Phil Hartman</th>
      <th>supporting_Philip Seymour Hoffman</th>
      <th>supporting_Phoebe Cates</th>
      <th>supporting_Pierce Brosnan</th>
      <th>supporting_Piper Laurie</th>
      <th>supporting_Piper Perabo</th>
      <th>supporting_Polly Adams</th>
      <th>supporting_Powers Boothe</th>
      <th>supporting_Priscilla Presley</th>
      <th>supporting_Queen Latifah</th>
      <th>supporting_Quentin Tarantino</th>
      <th>supporting_Quinton Aaron</th>
      <th>supporting_Rachel Griffiths</th>
      <th>supporting_Rachel McAdams</th>
      <th>supporting_Rachel Nichols</th>
      <th>supporting_Rachel Weisz</th>
      <th>supporting_Radha Mitchell</th>
      <th>supporting_Raini Rodriguez</th>
      <th>supporting_Ralph Fiennes</th>
      <th>supporting_Ralph Macchio</th>
      <th>supporting_Rami Malek</th>
      <th>supporting_Randy Quaid</th>
      <th>supporting_Raoul Bova</th>
      <th>supporting_Raquel Castro</th>
      <th>supporting_Raquel Welch</th>
      <th>supporting_Ray Allen</th>
      <th>supporting_Ray Liotta</th>
      <th>supporting_Ray Winstone</th>
      <th>supporting_Ray Wise</th>
      <th>supporting_Raymond J. Barry</th>
      <th>supporting_Rebecca Hall</th>
      <th>supporting_Reese Witherspoon</th>
      <th>supporting_Regina Hall</th>
      <th>supporting_Rei Sakuma</th>
      <th>supporting_Rene Russo</th>
      <th>supporting_Renée Zellweger</th>
      <th>supporting_Rhys Ifans</th>
      <th>supporting_Richard Attenborough</th>
      <th>supporting_Richard Basehart</th>
      <th>supporting_Richard Beymer</th>
      <th>supporting_Richard Burton</th>
      <th>supporting_Richard Dreyfuss</th>
      <th>supporting_Richard Gere</th>
      <th>supporting_Richard Kind</th>
      <th>supporting_Rick Moranis</th>
      <th>supporting_River Phoenix</th>
      <th>supporting_Rob Brown</th>
      <th>supporting_Robert Carlyle</th>
      <th>supporting_Robert De Niro</th>
      <th>supporting_Robert Downey Jr.</th>
      <th>supporting_Robert Duvall</th>
      <th>supporting_Robert Englund</th>
      <th>supporting_Robert Hoffman</th>
      <th>supporting_Robert Pattinson</th>
      <th>supporting_Robert Redford</th>
      <th>supporting_Robert Sean Leonard</th>
      <th>supporting_Robert Shaw</th>
      <th>supporting_Robert Walker</th>
      <th>supporting_Roberto Benigni</th>
      <th>supporting_Robin Shou</th>
      <th>supporting_Robin Tunney</th>
      <th>supporting_Robin Williams</th>
      <th>supporting_Robin Wright</th>
      <th>supporting_Rochelle Davis</th>
      <th>supporting_Rod Taylor</th>
      <th>supporting_Rodney Dangerfield</th>
      <th>supporting_Roger B. Smith</th>
      <th>supporting_Ron Howard</th>
      <th>supporting_Ron Perlman</th>
      <th>supporting_Ronee Blakley</th>
      <th>supporting_Rooney Mara</th>
      <th>supporting_Rory Cochrane</th>
      <th>supporting_Rosamund Pike</th>
      <th>supporting_Rosanna Arquette</th>
      <th>supporting_Rosario Dawson</th>
      <th>supporting_Rosario Flores</th>
      <th>supporting_Rose Byrne</th>
      <th>supporting_Rufus Sewell</th>
      <th>supporting_Rupert Grint</th>
      <th>supporting_Russell Brand</th>
      <th>supporting_Russell Crowe</th>
      <th>supporting_Rutger Hauer</th>
      <th>supporting_Ryan Gosling</th>
      <th>supporting_Ryan Guzman</th>
      <th>supporting_Ryan Reynolds</th>
      <th>supporting_Sacha Baron Cohen</th>
      <th>supporting_Saffron Burrows</th>
      <th>supporting_Sally Field</th>
      <th>supporting_Salma Hayek</th>
      <th>supporting_Sam Neill</th>
      <th>supporting_Sam Rockwell</th>
      <th>supporting_Sam Worthington</th>
      <th>supporting_Samaire Armstrong</th>
      <th>supporting_Samuel L. Jackson</th>
      <th>supporting_Sandra Bullock</th>
      <th>supporting_Sarah Berry</th>
      <th>supporting_Sarah Michelle Gellar</th>
      <th>supporting_Sarah Roemer</th>
      <th>supporting_Sarita Choudhury</th>
      <th>supporting_Scarlett Johansson</th>
      <th>supporting_Scott Bakula</th>
      <th>supporting_Scott Caan</th>
      <th>supporting_Scott MacDonald</th>
      <th>supporting_Scott Speedman</th>
      <th>supporting_Sean Bean</th>
      <th>supporting_Sean Connery</th>
      <th>supporting_Sean Penn</th>
      <th>supporting_Sean Young</th>
      <th>supporting_Seann William Scott</th>
      <th>supporting_Sebastian Koch</th>
      <th>supporting_Selma Blair</th>
      <th>supporting_Seth Green</th>
      <th>supporting_Seth Rogen</th>
      <th>supporting_Shailene Woodley</th>
      <th>supporting_Shane West</th>
      <th>supporting_Shannyn Sossamon</th>
      <th>supporting_Shantel VanSanten</th>
      <th>supporting_Sharon Stone</th>
      <th>supporting_Shawnee Smith</th>
      <th>supporting_Shelley Duvall</th>
      <th>supporting_Sheri Moon Zombie</th>
      <th>supporting_Shia LaBeouf</th>
      <th>supporting_Shirley MacLaine</th>
      <th>supporting_Shu Qi</th>
      <th>supporting_Sienna Guillory</th>
      <th>supporting_Sienna Miller</th>
      <th>supporting_Sigourney Weaver</th>
      <th>supporting_Simon Chandler</th>
      <th>supporting_Sofía Vergara</th>
      <th>supporting_Sondra Locke</th>
      <th>supporting_Sonja Smits</th>
      <th>supporting_Sophia Myles</th>
      <th>supporting_Sophie Marceau</th>
      <th>supporting_Soren Fulton</th>
      <th>supporting_Spencer Breslin</th>
      <th>supporting_Stacy Keach</th>
      <th>supporting_Stanley Tucci</th>
      <th>supporting_Stefanie Scott</th>
      <th>supporting_Stephen Baldwin</th>
      <th>supporting_Stephen Rea</th>
      <th>supporting_Sterling Holloway</th>
      <th>supporting_Steve Carell</th>
      <th>supporting_Steve Coogan</th>
      <th>supporting_Steve Martin</th>
      <th>supporting_Steve Zahn</th>
      <th>supporting_Steven Bauer</th>
      <th>supporting_Sue Lyon</th>
      <th>supporting_Sung Kang</th>
      <th>supporting_Susan George</th>
      <th>supporting_Susan Sarandon</th>
      <th>supporting_Sylvester Stallone</th>
      <th>supporting_Sô Yamamura</th>
      <th>supporting_T.R. Knight</th>
      <th>supporting_Takeshi Kaneshiro</th>
      <th>supporting_Talia Shire</th>
      <th>supporting_Tara Morice</th>
      <th>supporting_Taraji P. Henson</th>
      <th>supporting_Taylor Dooley</th>
      <th>supporting_Taylor Kitsch</th>
      <th>supporting_Taylor Momsen</th>
      <th>supporting_Tencho Gyalpo</th>
      <th>supporting_Terence Stamp</th>
      <th>supporting_Teresa Palmer</th>
      <th>supporting_Teri Polo</th>
      <th>supporting_Terrence Howard</th>
      <th>supporting_Terry Alexander</th>
      <th>supporting_Thandie Newton</th>
      <th>supporting_Theo James</th>
      <th>supporting_Thomas Haden Church</th>
      <th>supporting_Thomas Kretschmann</th>
      <th>supporting_Thora Birch</th>
      <th>supporting_Tim Allen</th>
      <th>supporting_Tim Matheson</th>
      <th>supporting_Tim Robbins</th>
      <th>supporting_Tim Roth</th>
      <th>supporting_Timothy Olyphant</th>
      <th>supporting_Tina Fey</th>
      <th>supporting_Tina Turner</th>
      <th>supporting_Tobey Maguire</th>
      <th>supporting_Toby Kebbell</th>
      <th>supporting_Tom Cruise</th>
      <th>supporting_Tom Glynn-Carney</th>
      <th>supporting_Tom Hanks</th>
      <th>supporting_Tom Hardy</th>
      <th>supporting_Tom Hulce</th>
      <th>supporting_Tom Sizemore</th>
      <th>supporting_Tom Wilkinson</th>
      <th>supporting_Tommy Lee Jones</th>
      <th>supporting_Toni Collette</th>
      <th>supporting_Tony Curtis</th>
      <th>supporting_Tony Revolori</th>
      <th>supporting_Tracy Morgan</th>
      <th>supporting_Tuesday Knight</th>
      <th>supporting_Tye Sheridan</th>
      <th>supporting_Tyler Hoechlin</th>
      <th>supporting_Tyrese Gibson</th>
      <th>supporting_Téa Leoni</th>
      <th>supporting_Uma Thurman</th>
      <th>supporting_Unax Ugalde</th>
      <th>supporting_Val Bettin</th>
      <th>supporting_Val Kilmer</th>
      <th>supporting_Vanessa Bauche</th>
      <th>supporting_Vanessa Lachey</th>
      <th>supporting_Vera Farmiga</th>
      <th>supporting_Vera Miles</th>
      <th>supporting_Verna Bloom</th>
      <th>supporting_Verna Felton</th>
      <th>supporting_Viggo Mortensen</th>
      <th>supporting_Vin Diesel</th>
      <th>supporting_Vince Vaughn</th>
      <th>supporting_Vincent Cassel</th>
      <th>supporting_Vincent D'Onofrio</th>
      <th>supporting_Vincent Piazza</th>
      <th>supporting_Ving Rhames</th>
      <th>supporting_Virginia Cherrill</th>
      <th>supporting_Virginia Madsen</th>
      <th>supporting_Vladimir Kulich</th>
      <th>supporting_Vladimir Menshov</th>
      <th>supporting_Walter Huston</th>
      <th>supporting_Wayne Newton</th>
      <th>supporting_Wendy Raquel Robinson</th>
      <th>supporting_Wentworth Miller</th>
      <th>supporting_Wesley Snipes</th>
      <th>supporting_Whitney Houston</th>
      <th>supporting_Will Ferrell</th>
      <th>supporting_Will Forte</th>
      <th>supporting_Will Sasso</th>
      <th>supporting_Will Smith</th>
      <th>supporting_Willem Dafoe</th>
      <th>supporting_William Atherton</th>
      <th>supporting_William Baldwin</th>
      <th>supporting_William Forsythe</th>
      <th>supporting_William H. Macy</th>
      <th>supporting_William Holden</th>
      <th>supporting_William Hurt</th>
      <th>supporting_Winona Ryder</th>
      <th>supporting_Woody Harrelson</th>
      <th>supporting_Yaphet Kotto</th>
      <th>supporting_Yasiin Bey</th>
      <th>supporting_Yuriko Ishida</th>
      <th>supporting_Zac Efron</th>
      <th>supporting_Zach Galifianakis</th>
      <th>supporting_Zachary Quinto</th>
      <th>supporting_Zoe Saldana</th>
      <th>supporting_Zooey Deschanel</th>
      <th>supporting_Zoë Bell</th>
      <th>supporting_Zuleikha Robinson</th>
      <th>supporting_Óscar Jaenada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story</td>
      <td>862</td>
      <td>30000000.0</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jumanji</td>
      <td>8844</td>
      <td>65000000.0</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Heat</td>
      <td>949</td>
      <td>60000000.0</td>
      <td>187436818.0</td>
      <td>170.0</td>
      <td>7.7</td>
      <td>1886.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sudden Death</td>
      <td>9091</td>
      <td>35000000.0</td>
      <td>64350171.0</td>
      <td>106.0</td>
      <td>5.5</td>
      <td>174.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GoldenEye</td>
      <td>710</td>
      <td>58000000.0</td>
      <td>352194034.0</td>
      <td>130.0</td>
      <td>6.6</td>
      <td>1194.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1847 columns</p>
</div>




```python
directors.drop_duplicates(inplace=True)
directors["value"] = 1
```


```python
unique_dir_names = list(set(directors.director))
master = pd.DataFrame(columns=unique_dir_names)

for n in list(set(directors.id)):
    t = directors[directors["id"] == n]
    pivoted = t.pivot(index="id", columns="director", values="value")
    master = master.append(pivoted)
```


```python
master.fillna(0, inplace=True)
master.reset_index(level=0, inplace=True)
```


```python
final = pd.merge(master, dummies, left_on = 'index', right_on = 'id')
```

Finally, I'll drop the first director as a reference, as well as the __index__ and __id__ columns used for joining. The director I am dropping is "Aaron Seltzer", which is worth noting for when I interpret the coefficients.


```python
final.drop(["id", "index", "Aaron Seltzer", "dir_count", "title"], axis=1, inplace=True)
final.drop_duplicates(inplace=True)
```


```python
master.to_csv("director_dummies.csv")
```

### Pre-processing
Since the final product of this modelling is to produce a Flask app that allows someone to input factors about a film before it's produced to get the rating and the revenue generated, some features will have to be dropped. For example, a user would not know the *vote_count* before the film is created. Some of the features I am removing are the most correlated with the response variable so I will be losing some of the predictive power.

I'll begin by defining my train/test split. Then I'll standardise the remaining predictive variables since it's good practice for working with linear regressions. I'll fit and transform the scaling on my training set and transform on my testing set. 


```python
X = final.drop(["vote_average", "revenue", "vote_count"], axis=1)
y = final.vote_average
```


```python
scaler = StandardScaler()
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=42)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
print("Length of training sets:",len(X_train), len(X_test))
print("Length of testing sets:",len(y_train), len(y_test))
```

    Length of training sets: 1296 556
    Length of testing sets: 1296 556


### Modelling

#### Baseline score
I need a baseline score with which to compare the scores for all my models moving forward. This score will represent the score one would get if they were just to predict the mean value for _y_. If my model outperforms this score, I know it is doing well.


```python
y_pred_mean = [y_train.mean()] * len(y_test)

print("Baseline score RMSE: ",'{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_mean))))
```

    Baseline score RMSE:  0.83


The baseline score is an RMSE of 0.83, meaning that the predicted score is within 0.83 of a mark out of 10. I'll use this is a benchmark as to the performance of my models.

Ultimately, I will be choosing the model that has the best cross-validated score of all as this will be the one that generalises well. I will also take into account the training and test scores, since these will indicate over or underfitting.

#### Function to generate model scores
Since I'll be trying out many different models, I'll build a function that returns all the information for efficiency. I'll create one function for simple models and another for models utilising regularisation.


```python
# Function to return simple model metrics
def get_model_metrics(X_train, y_train, X_test, y_test, model, parametric=True):
    """This function takes the train-test splits as arguments, as well as the algorithm 
    being used, and returns the training score, the test score (both RMSE), the 
    cross-validated scores and the mean cross-validated score. It also returns the appropriate 
    feature importances depending on whether the optional argument 'parametric' is equal to 
    True or False."""
    
    model.fit(X_train, y_train)
    train_pred = np.around(model.predict(X_train),1)
    test_pred = np.around(model.predict(X_test),1)
    
    print('Training score', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_train, train_pred))))
    print('Testing RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, test_pred))))
    cv_scores = -ms.cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('Cross-validated RMSEs:', np.sqrt(cv_scores))
    print('Mean cross-validated RMSE:', '{0:0.2f}'.format(np.sqrt(np.mean(cv_scores))))
    
    if parametric == True:
        print(pd.DataFrame(list(zip(X_train.columns, model.coef_, abs(model.coef_))), 
                 columns=['Feature', 'Coef', 'Abs Coef']).sort_values('Abs Coef', ascending=False).head(10))
    else:
        print(pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)), 
                 columns=['Feature', 'Importance']).sort_values('Importance', ascending=False).head(10))
    
    return model
```


```python
# Function to return regularised model metrics
def regularised_model_metrics(X_train, y_train, X_test, y_test, model, grid_params, parametric=True):
    """This function takes the train-test splits as arguments, as well as the algorithm being 
    used and the parameters, and returns the best cross-validated training score, the test 
    score, the best performing model and it's parameters, and the feature importances."""
    
    gridsearch = GridSearchCV(model,
                              grid_params,
                              n_jobs=-1, cv=5, verbose=1, error_score='neg_mean_squared_error')
    
    gridsearch.fit(X_train, y_train)
    print('Best parameters:', gridsearch.best_params_)
    print('Cross-validated score on test data:', '{0:0.2f}'.format(abs(gridsearch.best_score_)))
    best_model = gridsearch.best_estimator_
    print('Testing RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, best_model.predict(X_test)))))
    
    if parametric == True:
        print(pd.DataFrame(list(zip(X_train.columns, best_model.coef_, abs(best_model.coef_))), 
                 columns=['Feature', 'Coef', 'Abs Coef']).sort_values('Abs Coef', ascending=False).head(10))
    else:
        print(pd.DataFrame(list(zip(X_train.columns, best_model.feature_importances_)), 
                 columns=['Feature', 'Importance']).sort_values('Importance', ascending=False).head(10))
    
    return best_model
```

#### Linear regression
##### Simple


```python
lm_simple = get_model_metrics(X_train, y_train, X_test, y_test, lm.LinearRegression())
lm_simple
```

    Training score 0.16
    Testing RMSE: 28456150374210.60
    Cross-validated RMSEs: [8.93832333e+12 2.50381131e+13 3.51881686e+13 1.54939005e+13
     2.66140868e+13]
    Mean cross-validated RMSE: 24055679197777.84
                              Feature          Coef      Abs Coef
    287             lead_Aileen Quinn  1.236628e+13  1.236628e+13
    563             lead_J.J. Johnson  7.765516e+12  7.765516e+12
    98                Jason Friedberg -7.324091e+12  7.324091e+12
    33                Brian Helgeland -7.206760e+12  7.206760e+12
    595        lead_Jason Schwartzman -5.903357e+12  5.903357e+12
    1807   supporting_Olivia Williams  5.676954e+12  5.676954e+12
    276                    The Cartel  4.742263e+12  4.742263e+12
    44                 Clyde Geronimi -4.666503e+12  4.666503e+12
    1954  supporting_Shannyn Sossamon  4.071398e+12  4.071398e+12
    284             lead_Adam Sandler  4.013993e+12  4.013993e+12





    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



The test score and mean cross-validated scores are both terrible, way above baseline, and the OK training score indicates huge overfitting. This is not too surprising with such a simple model, so introducing some regularisation should help.

##### Regularised - Ridge
For every model I use regularisation on, I will get a list of the parameters so I can select which ones to gridsearch.


```python
ridge = lm.Ridge()
list(ridge.get_params())
```




    ['alpha',
     'copy_X',
     'fit_intercept',
     'max_iter',
     'normalize',
     'random_state',
     'solver',
     'tol']




```python
ridge_params = {'alpha': np.logspace(-10, 10, 10),
               'fit_intercept': [True, False],
               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

ridge_model = regularised_model_metrics(X_train, y_train, X_test, y_test, ridge, ridge_params)
```

    Fitting 5 folds for each of 140 candidates, totalling 700 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   34.3s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  3.3min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:  8.1min
    [Parallel(n_jobs=-1)]: Done 700 out of 700 | elapsed:  9.6min finished


    Best parameters: {'alpha': 2154.4346900318865, 'fit_intercept': True, 'solver': 'sparse_cg'}
    Cross-validated score on test data: 0.20
    Testing RMSE: 0.73
                          Feature      Coef  Abs Coef
    247                   runtime  0.057142  0.057142
    258                     Drama  0.035586  0.035586
    255                    Comedy -0.035077  0.035077
    246                    budget -0.034000  0.034000
    249                    Action -0.025765  0.025765
    25               Billy Wilder  0.024681  0.024681
    468         lead_Eddie Murphy -0.024559  0.024559
    187         Quentin Tarantino  0.024217  0.024217
    41          Christopher Nolan  0.024203  0.024203
    1594  supporting_Katie Holmes -0.021963  0.021963


The cross-validated training score and the test score here looks a lot more promising - although the model is overfitting, the test score is still better than baseline. The coefficients make a bit more sense, but I'll see if the other models reflect this.

##### Regularised - Lasso


```python
lasso = lm.Lasso()
list(lasso.get_params())
```




    ['alpha',
     'copy_X',
     'fit_intercept',
     'max_iter',
     'normalize',
     'positive',
     'precompute',
     'random_state',
     'selection',
     'tol',
     'warm_start']




```python
lasso_params = {'alpha': np.logspace(-10, 10, 10),
               'fit_intercept': [True, False]}

lasso_model = regularised_model_metrics(X_train, y_train, X_test, y_test, lasso, lasso_params)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   23.2s


    Best parameters: {'alpha': 0.07742636826811278, 'fit_intercept': True}
    Cross-validated score on test data: 0.18
    Testing RMSE: 0.75
                                Feature      Coef  Abs Coef
    247                         runtime  0.204818  0.204818
    246                          budget -0.100973  0.100973
    258                           Drama  0.048917  0.048917
    255                          Comedy -0.037615  0.037615
    249                          Action -0.034441  0.034441
    251                       Animation  0.028058  0.028058
    25                     Billy Wilder  0.024367  0.024367
    468               lead_Eddie Murphy -0.003947  0.003947
    41                Christopher Nolan  0.003654  0.003654
    1380  supporting_Herbert Grönemeyer  0.000000  0.000000


    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   27.5s finished


The cross-validated training score is better here, although the test score is worse. The coefficients are also similar to the simple linear regression. 

##### Regularised - ElasticNet


```python
elastic = lm.ElasticNet()
list(elastic.get_params())
```




    ['alpha',
     'copy_X',
     'fit_intercept',
     'l1_ratio',
     'max_iter',
     'normalize',
     'positive',
     'precompute',
     'random_state',
     'selection',
     'tol',
     'warm_start']




```python
elastic_params = {'alpha': np.logspace(-10, 10, 10),
                 'l1_ratio': np.linspace(0.05, 0.95, 10),
                 'fit_intercept': [True, False]}

elastic_model = regularised_model_metrics(X_train, y_train, X_test, y_test, elastic, 
                                          elastic_params)
```

    Fitting 5 folds for each of 200 candidates, totalling 1000 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   37.9s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:  4.6min
    [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed:  5.0min
    [Parallel(n_jobs=-1)]: Done 1000 out of 1000 | elapsed:  5.3min finished


    Best parameters: {'alpha': 0.07742636826811278, 'fit_intercept': True, 'l1_ratio': 0.25}
    Cross-validated score on test data: 0.29
    Testing RMSE: 0.68
                   Feature      Coef  Abs Coef
    247            runtime  0.218141  0.218141
    246             budget -0.162801  0.162801
    255             Comedy -0.086776  0.086776
    25        Billy Wilder  0.074144  0.074144
    251          Animation  0.069016  0.069016
    258              Drama  0.056174  0.056174
    41   Christopher Nolan  0.054816  0.054816
    249             Action -0.044436  0.044436
    239       Wes Anderson  0.044365  0.044365
    187  Quentin Tarantino  0.043444  0.043444


The best testing score yet, with less overfitting. All the linear regression models seem to have performed similarly. I'll see how other models perform now.

#### Tree models
##### Simple decision tree


```python
dt = get_model_metrics(X_train, y_train, X_test, y_test, tree.DecisionTreeRegressor(), 
                       parametric=False)
dt
```

    Training score 0.00
    Testing RMSE: 0.84
    Cross-validated RMSEs: [0.84988687 0.79734114 0.79138703 0.78999536 0.85444556]
    Mean cross-validated RMSE: 0.82
                       Feature  Importance
    247                runtime    0.214021
    246                 budget    0.212694
    258                  Drama    0.027402
    251              Animation    0.015236
    259                 Family    0.015190
    260                Fantasy    0.013420
    67                Eli Roth    0.011142
    250              Adventure    0.011036
    272        Science Fiction    0.010173
    248  belongs_to_collection    0.008883





    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')



I expected a single decision tree to overfit, and we can see here from the perfect test score that it has. The test score is above the baseline, so I wouldn't use this model.

##### Random forest


```python
rf = get_model_metrics(X_train, y_train, X_test, y_test, ensemble.RandomForestRegressor(), 
                       parametric=False)
rf
```

    Training score 0.27
    Testing RMSE: 0.70
    Cross-validated RMSEs: [0.65723342 0.62455729 0.64259098 0.67142504 0.66986168]
    Mean cross-validated RMSE: 0.65
                       Feature  Importance
    247                runtime    0.225637
    246                 budget    0.219449
    258                  Drama    0.014884
    249                 Action    0.013076
    255                 Comedy    0.012790
    251              Animation    0.011984
    250              Adventure    0.011676
    248  belongs_to_collection    0.011356
    259                 Family    0.010790
    272        Science Fiction    0.009512





    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)



I believe this to be the best model so far. The cross-validated score is good compared with the cross-validated scores returned from the other models.

##### Gridsearched Random forest


```python
rrf = ensemble.RandomForestRegressor()
list(rrf.get_params())
```




    ['bootstrap',
     'criterion',
     'max_depth',
     'max_features',
     'max_leaf_nodes',
     'min_impurity_decrease',
     'min_impurity_split',
     'min_samples_leaf',
     'min_samples_split',
     'min_weight_fraction_leaf',
     'n_estimators',
     'n_jobs',
     'oob_score',
     'random_state',
     'verbose',
     'warm_start']




```python
rrf_params = {'bootstrap': [True, False],
             'max_depth': np.linspace(5, 50, 5),
             'min_samples_split': np.linspace(0.01, 1, 5),
             'n_estimators': [40, 50, 60]}

rrf_model = regularised_model_metrics(X_train, y_train, X_test, y_test, rrf, rrf_params,
                                     parametric=False)
```

    Fitting 5 folds for each of 150 candidates, totalling 750 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   13.6s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:   57.2s
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=-1)]: Done 750 out of 750 | elapsed:  5.0min finished


    Best parameters: {'bootstrap': True, 'max_depth': 50.0, 'min_samples_split': 0.01, 'n_estimators': 50}
    Cross-validated score on test data: 0.34
    Testing RMSE: 0.68
                       Feature  Importance
    247                runtime    0.243007
    246                 budget    0.232748
    258                  Drama    0.018196
    251              Animation    0.014074
    249                 Action    0.011856
    264                 Horror    0.011726
    259                 Family    0.010169
    255                 Comedy    0.009754
    250              Adventure    0.009498
    248  belongs_to_collection    0.007684


I expected this model to give one of the best performances, so I am not surprised to see it give the best cross-validated score so far. Although the test score isn't too far off that of the simple random forest, this is the best model so far.

##### Bagged decision trees


```python
bagdt = ensemble.BaggingRegressor()
bagdt.fit(X_train, y_train)

print('Training RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_train, bagdt.predict(X_train)))))
print('Testing RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, bagdt.predict(X_test)))))
cv_scores = -ms.cross_val_score(bagdt, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print('Cross-validated RMSEs:', cv_scores)
print('Mean cross-validated RMSE:', '{0:0.2f}'.format(np.mean(cv_scores)))
```

    Training RMSE: 0.28
    Testing RMSE: 0.72
    Cross-validated RMSEs: [0.44400885 0.43938494 0.3979722  0.473639   0.46053475]
    Mean cross-validated RMSE: 0.44


This performs very well, although the gridsearched random forest outperforms it.

#### Support Vector Machine
##### LinearSVR


```python
lin = svm.LinearSVR() 

lin_params = {
    'C': np.logspace(-3, 2, 5),
    'loss': ['epsilon_insensitive','squared_epsilon_insensitive'],
    'fit_intercept': [True,False],
    'max_iter': [1000]
}

lin_model = regularised_model_metrics(X_train, y_train, X_test, y_test, lin, lin_params)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    9.9s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.5min finished


    Best parameters: {'C': 0.001, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'max_iter': 1000}
    Cross-validated score on test data: 76.13
    Testing RMSE: 5.23
                                  Feature          Coef      Abs Coef
    179                    Peter Farrelly -2.351553e-15  2.351553e-15
    793                  lead_Naomi Watts  1.901257e-15  1.901257e-15
    90                     Hayao Miyazaki -1.775381e-15  1.775381e-15
    52                      David Frankel  1.720209e-15  1.720209e-15
    27                     Bobby Farrelly  1.552279e-15  1.552279e-15
    92                        J.J. Abrams  1.449619e-15  1.449619e-15
    1148  supporting_Catherine Zeta-Jones  1.289997e-15  1.289997e-15
    1900        supporting_Robin Williams  1.251860e-15  1.251860e-15
    67                           Eli Roth  1.186930e-15  1.186930e-15
    102               Jean-Jacques Annaud  1.105900e-15  1.105900e-15


The scores here are terrible. This is my first foray into support vector machines, and I may not be optimising them correctly. However these scores plus the performance running them and the coefficients returned makes me feel they may not be the most appropriate models for this scenario.

##### RBF


```python
rbf = svm.SVR(kernel='rbf')

rbf_params = {
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-4, 1, 5),
    'kernel': ['rbf']}

rbf = GridSearchCV(rbf, rbf_params, n_jobs=-1, cv=5, verbose=1, error_score='neg_mean_squared_error')
rbf.fit(X_train, y_train)
print('Best parameters:', rbf.best_params_)
print('Cross-validated Training RMSE:', '{0:0.2f}'.format(abs(rbf.best_score_)))
print('Testing RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, rbf.best_estimator_.predict(X_test)))))
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 125 out of 125 | elapsed:  3.0min finished


    Best parameters: {'C': 1.0, 'gamma': 0.0001, 'kernel': 'rbf'}
    Cross-validated Training RMSE: 0.14
    Testing RMSE: 0.76


This gives by far the best cross-validated train score - although the test score isn't the best, the model obviously generalises well.

##### Poly


```python
poly = svm.SVR(kernel='poly')

poly_params = {
    'C': np.linspace(0.01, 0.2, 10),
    'gamma': np.logspace(-5, 2, 10),
    'degree': [2]}

poly = GridSearchCV(poly, poly_params, n_jobs=-1, cv=5, verbose=1, error_score='neg_mean_squared_error')
poly.fit(X_train, y_train)
print('Best parameters:', poly.best_params_)
print('Cross-validated Training RMSE:', '{0:0.2f}'.format(abs(poly.best_score_)))
print('Testing RMSE:', '{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, poly.best_estimator_.predict(X_test)))))
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  4.4min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 10.2min
    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed: 11.5min finished


    Best parameters: {'C': 0.09444444444444444, 'degree': 2, 'gamma': 0.01291549665014884}
    Cross-validated Training RMSE: 0.08
    Testing RMSE: 0.80


This model has the best cross-validated train score, so although the test score isn't the greatest, it's the one I will select for my Flask app.

### Conclusion
Now the plan is:
* Export the scaled dataset as a csv to be used in the Flask app
* Pickle the SVR model to use in the app


```python
X_scaled = pd.concat([X_train, X_test])
y_concat = pd.concat([y_train, y_test])
```


```python
X_scaled.to_csv("X_ratings.csv")
```


```python
final_model = poly.best_estimator_
final_model.fit(X_scaled, y_concat)
cv_scores = -ms.cross_val_score(final_model, X_scaled, y_concat, cv=5, scoring='neg_mean_squared_error')
print('Cross-validated RMSEs:', np.sqrt(cv_scores))
print('Mean cross-validated RMSE:', '{0:0.2f}'.format(np.mean(np.sqrt(cv_scores))))
```

    Cross-validated RMSEs: [0.74717048 0.75255535 0.76234458 0.73663802 0.78108789]
    Mean cross-validated RMSE: 0.76



```python
with open('model_ratings.pkl', 'wb') as f:
    pickle.dump(final_model, f)
```
