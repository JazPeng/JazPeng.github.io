---
layout: default
title:  "Predicting IMDB ratings for movies"
permalink: "/predict_movie_ratings/"
---

# <center>Predicting IMDB ratings for movies</center>
## <center>Python</center>

I consider myself an enthusiastic, if not particularly knowledgable, cinephile. When debating which movie to watch out of a selection of a few, my partner and I always do the "IMDB test" - the movie we choose is the one with the highest average rating on IMDB.

I would like to see how easy it could be to predict the average rating for a model, and what predictors have the most effect.

The dataset I used is the popular [Movies dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) found on Kaggle. The aim is to try different models and find the one with the lowest root mean squared error (RMSE).

Once I have identified the best performing model, I will create an interactive interface to allow users to predict the average rating of movies with certain feature inputs (such as cast, crew and production company).


```python
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import linear_model as lm, metrics, tree, ensemble, model_selection as ms

%matplotlib inline

pd.options.mode.chained_assignment = None

np.random.seed(42)
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

### Credits


```python
credits = pd.read_csv("/Users/jasminepengelly/Desktop/movie_revenue_predictor/credits.csv")
```


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



### Movies metadata
There are a lot of data in this csv that I do not consider relevant to use for my features. I will have to extract the data I want to use.


```python
movies_metadata = pd.read_csv("/Users/jasminepengelly/Desktop/movie_revenue_predictor/movies_metadata.csv", low_memory = False)
```


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



Now I'll work on getting the *genres* csv into a workable format.


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



In order to add the appropriate genres to the *metadata* dataframe, I need to create a pivot table that indicates which film *id* is which genre. I will then merge the flattened pivot table with the original *metadata* dataframe.


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



Now add the cast to the main dataframe, selecting the lead and the supporting actor as the two features.


```python
lead = cast[cast['order'] == 0]
lead = lead[['movie_id', 'name']]
lead = lead.rename(columns={'movie_id':'id'})
lead = lead.rename(columns={'name':'lead'})
lead.head()
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
    <tr>
      <th>39</th>
      <td>15602</td>
      <td>Walter Matthau</td>
    </tr>
    <tr>
      <th>46</th>
      <td>31357</td>
      <td>Whitney Houston</td>
    </tr>
    <tr>
      <th>56</th>
      <td>11862</td>
      <td>Steve Martin</td>
    </tr>
  </tbody>
</table>
</div>




```python
metadata_genre.dropna(inplace = True)
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
dataset.dropna(inplace = True)
```

Time for some cleaning. I remove all rows with null revenue and budget and where the vote count was below 50. I'll also convert each column to the appropriate data type and delete the duplicates.


```python
final_dataset = dataset.loc[(dataset['budget'] != '0') & (dataset['revenue'] != 0)]
final_dataset['budget'] = final_dataset['budget'].astype('float64')
final_dataset['belongs_to_collection'] = final_dataset['belongs_to_collection'].astype('int64')
final_dataset = final_dataset[final_dataset['vote_count'] > 50]
```


```python
len(final_dataset)
```




    4676




```python
final_dataset.drop_duplicates(inplace=True)
len(final_dataset)
```




    4632




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
Though I managed to drop the duplicates above, I can see that some films have multiple directors. This leaves duplicated films that weren't removed with the above because the *director* columns are different.

My intention was originally to One-Hot Encode the directors and merge them into the original data frame. However, the data set became too large to process on my laptop so I'll try again but with the least frequent directors removed to make the dataset more manageable.


```python
final_dataset['dir_count'] = final_dataset.groupby('director')['director'].transform('count')
final_dataset = final_dataset[final_dataset["dir_count"] >= 5]
final_dataset_wo_dir = final_dataset.drop("director", axis=1).drop_duplicates(keep="first")
directors = final_dataset[["id", "director"]]
```

I'll now export the *final_dataset_wo_dir* for use in my next analysis.


```python
final_dataset_wo_dir.to_csv("movies_wo_dir.csv")
```

## Exploratory data analysis
Now it's time to look and at all of my features and look out for things like multicollinearity that could affect my model.

First, I'll define my predictors (without *director*) and response variables.


```python
X = ['id', 'budget', 'runtime', 'vote_count', 'belongs_to_collection', 'Action', 'Adventure',
              'Animation', 'Aniplex', 'BROSTA TV', 'Carousel Productions', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Family', 'Fantasy', 'Foreign', 'GoHands', 'History', 'Horror', 'Mardock Scramble Production Committee',
              'Music', 'Mystery', 'Odyssey Media', 'Pulser Productions', 'Rogue State', 'Romance', 'Science Fiction',
              'Sentai Filmworks', 'TV Movie', 'Telescene Film Group Productions', 'The Cartel', 'Thriller',
              'Vision View Entertainment', 'War', 'Western', 'lead', 'supporting', 'revenue']

y = 'vote_average'
```

Now I'll check for correlation between the predictors.


```python
sns.heatmap(final_dataset_wo_dir[X].corr(), vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(10, 220, sep=80, n=7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a412f3d68>




![Correlation](https://raw.githubusercontent.com/JazPeng/assets/master/movies/rate_corr.png)


There are four stronger correlations I need to take into account when looking at my features:

* _revenue_ and _budget_
* *vote\_count* and _budget_
* _revenue_ and *vote\_count*
* _family_ and _animation_

I'll look at these relationships in greater detail below.

#### *revenue* and *budget*


```python
sns.jointplot(x='revenue', y='budget', data=final_dataset_wo_dir)
```




    <seaborn.axisgrid.JointGrid at 0x1a44080f60>




![Budget vs Revenue](https://raw.githubusercontent.com/JazPeng/assets/master/movies/budget_vs_revenue.png)


The correlation between *revenue* and *budget* is 0.69 which a strong enough correlation to consider PCA. It also makes intuitive sense that movies that have larger budgets will attract larger audiences.

#### *revenue* and *vote_count*


```python
sns.jointplot(x='revenue', y='vote_count', data=final_dataset_wo_dir)
```




    <seaborn.axisgrid.JointGrid at 0x119d78d30>




![Revenue vs vote count](https://raw.githubusercontent.com/JazPeng/assets/master/movies/vote_count_vs_revenue.png)


The correlation between *revenue* and *vote_count* is even more significant at 0.74. This is further evidence for PCA. As with the relationship between *revenue* and *budget*, the larger an audience the higher the number of votes can be expected.

#### *vote\_count* and *budget*


```python
sns.jointplot(x='budget', y='vote_count', data=final_dataset_wo_dir)
```




    <seaborn.axisgrid.JointGrid at 0x11a6e0828>




![Budget vs vote count](https://raw.githubusercontent.com/JazPeng/assets/master/movies/budget_vs_votecount.png)


This relationship alone, with a correlation coefficient of 0.52, is not enough to require PCA. however, these variables' relationship with *revenue* makes me believe reducing these three variables into two or one is necessary.

#### *Family* and *Animation*


```python
sns.jointplot(x='Family', y='Animation', data=final_dataset_wo_dir)
```




    <seaborn.axisgrid.JointGrid at 0x11c6aa240>




![Family vs animation](https://raw.githubusercontent.com/JazPeng/assets/master/movies/animation_vs_family.png)


Not only is the correlation here too low to give much thought to, but the relationship isn't linear. I won't worry about this relationship moving forward.

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
As well as making the variables in the *final_dataset_wo_dir* dummy, I also need to manually add the directors. I'll do this by creating a pivot table with the directors as columns, the movie id as the row and the values as either 0 or 1.


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
      <th>lead_Alexander SkarsgÃ¥rd</th>
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
      <th>lead_Andy GarcÃ­a</th>
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
      <th>lead_BjÃ¶rk</th>
      <th>lead_Blake Jenner</th>
      <th>lead_Blake Lively</th>
      <th>lead_Bob Hoskins</th>
      <th>lead_Bob Newhart</th>
      <th>lead_Bobby Campo</th>
      <th>lead_Bobby Driscoll</th>
      <th>lead_Bodil JÃžrgensen</th>
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
      <th>lead_Daniel BrÃŒhl</th>
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
      <th>lead_DemiÃ¡n Bichir</th>
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
      <th>supporting_RenÃ©e Zellweger</th>
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
      <th>supporting_SofÃ­a Vergara</th>
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
      <th>supporting_SÃŽ Yamamura</th>
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
      <th>supporting_TÃ©a Leoni</th>
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
      <th>supporting_ZoÃ« Bell</th>
      <th>supporting_Zuleikha Robinson</th>
      <th>supporting_Ãscar Jaenada</th>
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
<p>5 rows Ã 1847 columns</p>
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

Finally, I'll drop the first director as a reference, as well as the *index* and *id* columns used for joining.


```python
final.drop(["id", "index", "Aaron Seltzer", "dir_count", "title"], axis=1, inplace=True)
final.drop_duplicates(inplace=True)
```


```python
master.to_csv("director_dummies.csv")
```

### Principal component analysis
I will work exclusively with the three correlated variables - *revenue*, *vote_count* and *budget*. I will reduce them to two and three principal components and see which keeps the most information.


```python
from sklearn.decomposition import PCA
```


```python
var_to_pca = final[["budget", "vote_count", "revenue"]]
```


```python
for n in (1, 2):
    pca = PCA(n_components=n)
    points = pca.fit_transform(var_to_pca)
    print("Variance with ",n,":",pca.explained_variance_ratio_)
    print("Information kept with ",n,":",sum(pca.explained_variance_ratio_))
    print("\n")
```

    Variance with  1 : [ 0.97421975]
    Information kept with  1 : 0.974219745908


    Variance with  2 : [ 0.97421975  0.02578025]
    Information kept with  2 : 0.999999999976




Reducing the three correlated variables to two has kept 99%+ of the information. This is great, but it seems to make sense to use the one variable which retains 97%+ of the information by itself.

I'll use this one feature for my models, dropping the three features it was made up of.


```python
pca = PCA(n_components=1)
b_v_r = pca.fit_transform(var_to_pca)
```


```python
final["b_v_r"] = b_v_r
final.drop(["budget", "vote_count", "revenue"], axis=1, inplace=True)
```

## Modelling

I'll begin by redefining my variables.


```python
X = final.drop("vote_average", axis=1)
y = final.vote_average
```


```python
from sklearn.model_selection import train_test_split, KFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train), len(X_test))
print(len(y_train), len(y_test))
```

    1296 556
    1296 556


### Dumb model
First, I'll create a dumb model to compare the results of all my linear regression models with. This dumb model will try and predict the mean value, if the errors for my models are lower I know they are performing somewhat OK.


```python
y_pred_mean = [y_train.mean()] * len(y_test)

print("Dumb model RMSE: ",'{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_mean))))
```

    Dumb model RMSE:  0.83


The dumb model has an RMSE of 0.83. I'll use this is a benchmark as to the performance of my models.

### Linear regression
I will begin with a standard linear regression model and gradually make it more complex with cross-validation and regularisation. I will compare the accuracy of all the models and select the best version before I move onto other regression models.

#### Function to get relevant metrics
For reproducibility and ease, I'll create a function that takes the linear regression model, the predictive variables and the response variable as an input and returns the relative graphs and metrics.


```python
def get_linear_model_metrics(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    residuals = (y_train - model.predict(X_train)).values
    print('Coefficients:', model.coef_)
    print('y-intercept:', model.intercept_)
    print('R-Squared:', model.score(X_train, y_train))
    print("Training error (RMSE): ",'{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_train, residuals))))
    print("Testing error (RMSE): ",'{0:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, lm.predict(X_test)))))

    plt.figure()
    plt.hist(residuals)
    return model
```

#### Standard linear regression model


```python
from sklearn import feature_selection, linear_model
```


```python
lm = linear_model.LinearRegression()

get_linear_model_metrics(X_train, y_train, X_test, y_test, lm)
```

    Coefficients: [  1.97964001e+03  -5.28181540e+00   4.98469233e+01 ...,   0.00000000e+00
       5.24458970e+01   1.91766958e-09]
    y-intercept: -55.8755805019
    R-Squared: 0.994974678012
    Training error (RMSE):  6.57
    Testing error (RMSE):  165.97





    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




![Residuals](https://raw.githubusercontent.com/JazPeng/assets/master/movies/rate_lr.png)


The R-Squared is high which is good, although this isn't always the best indicator of a model. The training error isn't great - being wrong by 6.57 points per on average out of a score of 10 is bad. The testing error is much higher, which indicates a huge amount of overfitting.

To counteract this, I will now try various cross-validated models with *k* ranging between 5 and 10.

#### Cross-validated linear regression model


```python
for k in range(5, 11):
    cv_scores = cross_val_score(linear_model.LinearRegression(), X_train, y_train, scoring='neg_mean_squared_error', cv=k)
    print("Scores for k =",k,":",list(map(lambda score: '{0:0.2f}'.format(np.sqrt(-score)), cv_scores)))
    print("Variance: ", np.var(cv_scores))
    print("Mean RMSE for k =",k,":",'{0:0.2f}'.format(np.mean(np.sqrt(-cv_scores))))
    print("\n")
```

    Scores for k = 5 : ['379.40', '290.56', '282.81', '672.45', '41.89']
    Variance:  24501055596.5
    Mean RMSE for k = 5 : 333.42


    Scores for k = 6 : ['521.96', '414.68', '254.82', '340.78', '130.53', '762.90']
    Variance:  35103443270.7
    Mean RMSE for k = 6 : 404.28


    Scores for k = 7 : ['636.83', '478.56', '34.64', '25.25', '76.22', '286.50', '34.83']
    Variance:  21214458776.9
    Mean RMSE for k = 7 : 224.69


    Scores for k = 8 : ['737.89', '867.58', '56.91', '387.47', '229.41', '1.97', '322.37', '99.76']
    Variance:  71545681113.7
    Mean RMSE for k = 8 : 337.92


    Scores for k = 9 : ['1.95', '397.79', '14.40', '40.09', '105.34', '146.37', '735.20', '352.24', '42.90']
    Variance:  27911051299.3
    Mean RMSE for k = 9 : 204.03


    Scores for k = 10 : ['1.97', '741.07', '655.76', '197.92', '419.68', '104.16', '272.64', '468.92', '819.50', '244.88']
    Variance:  52989341088.0
    Mean RMSE for k = 10 : 392.65




The model with the lowest mean RMSE is *k* = 9.

While this model is the best performing, it's still dramatically under-performing comparative to the dumb model. Next step is to use GridSearch with both Ridge and Lasso, and cross-validate those models with *k* = 9.

#### Regularisation with cross-validation


```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
```

##### Ridge


```python
grid_ridge = GridSearchCV(estimator=Ridge(),
                    param_grid={'alpha': np.logspace(-10, 10, 21)},
                    scoring='neg_mean_squared_error',
                    return_train_score=True,
                    cv=9)

grid_ridge.fit(X_train, y_train)
```




    GridSearchCV(cv=9, error_score='raise',
           estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'alpha': array([  1.00000e-10,   1.00000e-09,   1.00000e-08,   1.00000e-07,
             1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03,   1.00000e+04,   1.00000e+05,
             1.00000e+06,   1.00000e+07,   1.00000e+08,   1.00000e+09,
             1.00000e+10])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='neg_mean_squared_error', verbose=0)




```python
print(np.sqrt(-grid_ridge.best_score_), grid_ridge.best_params_)
```

    0.752904014586 {'alpha': 100000.0}



```python
best_model_ridge = grid_ridge.best_estimator_
print("grid_ridge training RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_train, best_model_ridge.predict(X_train)))))
print("grid_ridge testing RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_test, best_model_ridge.predict(X_test)))))
```

    grid_ridge training RMSE: 0.75
    grid_ridge testing RMSE: 0.79


Regularisation has drastically improved the RMSE. The *best_model_ridge* has a testing and a training RMSE of 0.75. The testing error implies there is minimal overfitting, probably down to the cross-validation.

##### Lasso


```python
grid_lasso = GridSearchCV(estimator=Lasso(),
                    param_grid={'alpha': np.logspace(-10, 10, 21)},
                    scoring='neg_mean_squared_error',
                    return_train_score=True,
                    cv=9)

grid_lasso.fit(X_train, y_train)
```




    GridSearchCV(cv=9, error_score='raise',
           estimator=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'alpha': array([  1.00000e-10,   1.00000e-09,   1.00000e-08,   1.00000e-07,
             1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03,   1.00000e+04,   1.00000e+05,
             1.00000e+06,   1.00000e+07,   1.00000e+08,   1.00000e+09,
             1.00000e+10])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='neg_mean_squared_error', verbose=0)




```python
print(np.sqrt(-grid_lasso.best_score_), grid_lasso.best_params_)
```

    0.650936702899 {'alpha': 0.001}



```python
best_model_lasso = grid_lasso.best_estimator_
print("grid_lasso training RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_train, best_model_lasso.predict(X_train)))))
print("grid_lasso testing RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_test, best_model_lasso.predict(X_test)))))
```

    grid_lasso training RMSE: 0.53
    grid_lasso testing RMSE: 0.69


The *best_model_lasso* performed even better, with a testing RSME of 0.53. Though overfitting is implied in the variance between the training and testing score, the testing error still came in below the *best_model_ridge*. This far, this is the best performing model and outperforms the dumb model.

I will now move on to tree models.


### Decision tree
To find the best parameters for *min_samples_leaf* and *max_depth*, I will use GridSearch along with a 9-fold cross-validation.


```python
from sklearn.tree import DecisionTreeRegressor
```


```python
min_samples_leaf = list(range(1, 11))
max_depth_range = list(range(1, 11))
```


```python
grid_dt = GridSearchCV(estimator=DecisionTreeRegressor(),
                    param_grid={"min_samples_leaf": min_samples_leaf,
                                "max_depth": max_depth_range},
                    scoring="neg_mean_squared_error",
                    cv=9)

grid_dt.fit(X_train, y_train)
```




    GridSearchCV(cv=9, error_score='raise',
           estimator=DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best'),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='neg_mean_squared_error', verbose=0)




```python
print(np.sqrt(-grid_dt.best_score_), grid_dt.best_params_)
```

    0.72585231241 {'max_depth': 6, 'min_samples_leaf': 2}



```python
best_model_dt = grid_dt.best_estimator_
print("grid_dt training RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_train, best_model_dt.predict(X_train)))))
print("grid_dt testing RMSE: "'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_test, best_model_dt.predict(X_test)))))
```

    grid_dt training RMSE: 0.64
    grid_dt testing RMSE: 0.79


I would expect this model to overfit since it's a decision tree with no cross-validation, although the difference between the training error and the testing error isn't too large. The *best_model_lasso* is still the best performing.

Now to check out the feature importances according to the best performing tree.


```python
pd.DataFrame({'feature':X_train.columns, 'coefficients':best_model_dt.feature_importances_}).sort_values(by='coefficients', axis=0, ascending=False).head(10)
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
      <th>coefficients</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>246</th>
      <td>0.385262</td>
      <td>runtime</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.110519</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>2086</th>
      <td>0.095718</td>
      <td>b_v_r</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.063697</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.059005</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0.046741</td>
      <td>Family</td>
    </tr>
    <tr>
      <th>111</th>
      <td>0.036536</td>
      <td>Joel Schumacher</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.030459</td>
      <td>Billy Wilder</td>
    </tr>
    <tr>
      <th>244</th>
      <td>0.027942</td>
      <td>Woody Allen</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.021587</td>
      <td>Alfred Hitchcock</td>
    </tr>
  </tbody>
</table>
</div>



It looks like the most important features in defining how a movie will be rated is *runtime* by a significant amount. This seems odd (perhaps better rated films are longer?). It's followed by the genre *Drama* and then the principal component that combined *budget*, *vote_count* and *revenue*.

I'll do the same thing with the other tree models to see how they compare.

### Bagged decision trees


```python
from sklearn.ensemble import BaggingRegressor
```


```python
est = list(range(10, 110, 10))

for n in est:
    bag_dt = BaggingRegressor(DecisionTreeRegressor(), n_estimators=n, bootstrap=True, oob_score=True, random_state=42)
    score = cross_val_score(bag_dt, X_train, y_train, cv=9, scoring='neg_mean_squared_error')
    bag_dt.fit(X_train, y_train)
    print("Cross-validated score for bag_dt",n,": ",'{0:0.2f}'.format(np.mean(np.sqrt(-score))))
    print("bag_dt testing RMSE for",n,": ",'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_test, bag_dt.predict(X_test)))))
    print("\n")    
```

    Cross-validated score for bag_dt 10 :  0.70
    bag_dt testing RMSE for 10 :  0.74


    Cross-validated score for bag_dt 20 :  0.69
    bag_dt testing RMSE for 20 :  0.73


    Cross-validated score for bag_dt 30 :  0.68
    bag_dt testing RMSE for 30 :  0.72


    Cross-validated score for bag_dt 40 :  0.68
    bag_dt testing RMSE for 40 :  0.72


    Cross-validated score for bag_dt 50 :  0.68
    bag_dt testing RMSE for 50 :  0.72


    Cross-validated score for bag_dt 60 :  0.68
    bag_dt testing RMSE for 60 :  0.72


    Cross-validated score for bag_dt 70 :  0.68
    bag_dt testing RMSE for 70 :  0.72


    Cross-validated score for bag_dt 80 :  0.68
    bag_dt testing RMSE for 80 :  0.72


    Cross-validated score for bag_dt 90 :  0.68
    bag_dt testing RMSE for 90 :  0.72


    Cross-validated score for bag_dt 100 :  0.68
    bag_dt testing RMSE for 100 :  0.72




It seems there are no differences between bagged trees with between 30 and 100 estimators. Finally I'll move on to a random forest model.

### Random forest


```python
from sklearn.ensemble import RandomForestRegressor
```


```python
for n in est:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    score = cross_val_score(rf, X_train, y_train, cv=9, scoring='neg_mean_squared_error')
    rf.fit(X_train, y_train)
    print("rf training RMSE for",n,": ",'{0:0.2f}'.format(np.mean(np.sqrt(-score))))
    print("rf testing RMSE for",n,": ",'{0:0.2f}'.format(np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))))
    print("\n")
```

    rf training RMSE for 10 :  0.70
    rf testing RMSE for 10 :  0.74


    rf training RMSE for 20 :  0.69
    rf testing RMSE for 20 :  0.73


    rf training RMSE for 30 :  0.68
    rf testing RMSE for 30 :  0.73


    rf training RMSE for 40 :  0.68
    rf testing RMSE for 40 :  0.72


    rf training RMSE for 50 :  0.68
    rf testing RMSE for 50 :  0.72


    rf training RMSE for 60 :  0.68
    rf testing RMSE for 60 :  0.72


    rf training RMSE for 70 :  0.68
    rf testing RMSE for 70 :  0.72


    rf training RMSE for 80 :  0.68
    rf testing RMSE for 80 :  0.72


    rf training RMSE for 90 :  0.68
    rf testing RMSE for 90 :  0.72


    rf training RMSE for 100 :  0.68
    rf testing RMSE for 100 :  0.72




Unsurprisingly, the *rf* model's RMSE doesn't change between 30 and 100 restimators. I'll now check out the feature importances.


```python
rf = RandomForestRegressor(n_estimators=30, random_state=42)
rf.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
               oob_score=False, random_state=42, verbose=0, warm_start=False)




```python
pd.DataFrame({'feature':X_train.columns, 'coefficients':rf.feature_importances_}).sort_values(by='coefficients', axis=0, ascending=False).head(10)
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
      <th>coefficients</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>246</th>
      <td>0.191308</td>
      <td>runtime</td>
    </tr>
    <tr>
      <th>2086</th>
      <td>0.128969</td>
      <td>b_v_r</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0.036736</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>248</th>
      <td>0.026470</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>254</th>
      <td>0.016413</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.015228</td>
      <td>Billy Wilder</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0.014520</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0.013791</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0.010906</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0.008982</td>
      <td>Thriller</td>
    </tr>
  </tbody>
</table>
</div>



Again, *runtime* is the most important feature, and the principal component is also important. The genres of *Drama*, *Action* and *Comedy* are also important.

## Conclusion
The best performing model was the *best_model_lasso*, which is the one I will use behind my interactive interface. I'll run the same analysis with *revenue* and see which model best predicts that.
