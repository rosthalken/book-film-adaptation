# Import dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer as CV
import numpy as np


# Load in reviews
path_parent = os.path.dirname(os.getcwd())
path_to_reviews = os.path.join(path_parent, 'data', 'scraped_data')
reviews_df = pd.read_csv(os.path.join(path_to_reviews, 'combined_scraped_reviews.csv'))

# Isolate book reviews to those whose films were released after Goodreads created in 2006       
metadata_df = pd.read_csv(os.path.join(path_parent, 'data', 'connected_metadata.csv'))

# Get IDs of books that have any film released after 2009
filtered_df = metadata_df[metadata_df["film_year"] > 2009]
recent_film_ids = list(filtered_df["id"])

# in reviews_df, keep only rows where "book_id" is a number in the recent_film_ids list
def check_cell_value(cell):
  if cell in recent_film_ids:
    return "KEEP"
  else:
    return "DROP"

reviews_df["to_do"] = reviews_df["book_id"].apply(check_cell_value)
keep_reviews = reviews_df[reviews_df["to_do"] == "KEEP"]

# connect movie release year to keep_reviews
# merge keep_reviews and metadata_df on "id" and "book_id" keeping only the "film_year" col of metadata_df
reduced_metadata = filtered_df[["id", "film_year"]]
new_df = keep_reviews.merge(reduced_metadata, left_on = "book_id", right_on = "id")

# Create new column "release_time" with "before" or "after" based on when the review was added
# new_df["date"] is date of review
# new_df["film_year"] is date of release

new_df = new_df[new_df['date'].notna()]
new_df = new_df[new_df['film_year'].notna()]
new_df = new_df[new_df['text'].notna()]



def timing_of_review(df_row):
  from dateutil.parser import parse
  review_year = parse(str(df_row["date"]), fuzzy=True).year
  if review_year < df_row["film_year"]:
    timing = "before"
  else:
    timing = "after"
  return timing

new_df["timing"] = new_df.apply(timing_of_review, axis = 1)

from collections import Counter
print("Number of reviews before film release: ", Counter(new_df["timing"])["before"])
print("Number of reviews after (or in same year as) film release: ", Counter(new_df["timing"])["after"])

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
exclude = set(string.punctuation)

from adaptation_functions import basic_sanitize, bayes_compare_language

before_reviews = new_df[new_df["timing"] == "before"].text.to_list()
after_reviews = new_df[new_df["timing"] == "after"].text.to_list()

output_path = os.path.join(path_parent, 'output', 'figures')

z_scores, vocabulary = bayes_compare_language(before_reviews, after_reviews, output_path, exclude, stopwords)

# strip character names? with NER
# add informative prior
 