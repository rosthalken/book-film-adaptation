import pandas as pd
import os
import numpy as np
import json

input_path = os.getcwd()
file_name = 'all_reviews.csv'
path_to_reviews = os.path.join(input_path, 'output', file_name)

reviews_df = pd.read_csv(path_to_reviews)

book_ids_path = os.path.join(input_path, 'data', 'all_ids.txt')
with open(book_ids_path) as file:
    lines = file.readlines()
    ids = [line.rstrip() for line in lines]

# which reviews_df["book_id"] are not in ids? these are unsuccessful scraped books
unmatched_titles = np.setdiff1d(ids,reviews_df["book_id"],assume_unique=False)

# number of users
len(reviews_df["user_url"].unique())

# percent na text reviews
na_value_reviews = reviews_df["text"].isna().sum()
total_reviews = len(reviews_df["text"])
total_reviews - na_value_reviews
na_value_reviews / total_reviews * 100

reviews_df["book_id"].value_counts().describe()

reviews_df.boxplot(column=['rating'])

smaller_df = reviews_df.sample(200)

### NOW imdb data
# https://github.com/dojutsu-user/IMDB-Scraper
# Cite this paper: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib

imdb_file_name = 'imdb_dataset.json'
imdb_path = os.path.join(input_path, 'data', imdb_file_name)

with open(imdb_path, 'r') as j:
     imdb_content = json.loads(j.read())

imdb_df = pd.DataFrame(imdb_content)


# book to film metadata
book_to_film_file_name = 'metadata_book_film.csv'
book_to_film_path = os.path.join(input_path, 'data', book_to_film_file_name)
book_to_film_df = pd.read_csv(book_to_film_path)

# join imdb_df to book_to_film_df WHERE book_to_film_df's "film_title" == imdb_df's "title" 
# AND book_to_film_df's "film_year" == imdb_df's "year"
merged_df = pd.merge(book_to_film_df, imdb_df,  how='left', left_on=['film_title','film_year'], right_on = ['title','year'])

# reduce to no na for match:
connected_titles_df = merged_df[merged_df['title'].notna()]
len(new_df["book_title"].unique())

connected_titles_df.to_csv(os.path.join(input_path, 'data', 'connected_titles.csv'))