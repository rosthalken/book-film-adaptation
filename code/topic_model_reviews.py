from collections import defaultdict
from datetime import datetime
import math
from operator import itemgetter
import os
import random
import re

import numpy as np
import pandas as pd

import little_mallet_wrapper as lmw

path_parent = os.path.dirname(os.getcwd())
path_to_reviews = os.path.join(path_parent, 'data', 'scraped_data')
folders = os.listdir(path_to_reviews)

file_name = 'all_reviews.csv'
big_df_list = []

for folder in folders:
    if "scraped" in str(folder):
        path = os.path.join(path_to_reviews, folder, 'all_reviews.csv')
        review_df = pd.read_csv(path)
        big_df_list.append(review_df)

reviews_df = pd.concat(big_df_list)

# reviews_df.to_csv(os.path.join(path_to_reviews, 'combined_scraped_reviews.csv'))
reviews_df = pd.read_csv(os.path.join(path_to_reviews, 'combined_scraped_reviews.csv'))
        

path_to_mallet = "/Users/rosamondthalken/Applications/Mallet-Dev/bin/mallet"

reviews_df["text"] = reviews_df["text"].astype(str)

# reduce to only non-nan values:
reviews_df = reviews_df[reviews_df['text'] != "nan"]


# Using Maria Antoniak's little mallet wrapper (plus code from the lmw demo!)
training_data = [lmw.process_string(t) for t in reviews_df['text'].tolist()]
#training_data = [d for d in training_data if d.strip()]

len(training_data)

book_ratings = reviews_df["rating"].to_list()
len(book_ratings)

# remember, these are slightly duplicated
doc_ids = reviews_df["review_id"].to_list()

lmw.print_dataset_stats(training_data)


### TOPIC MODEL ###
num_topics = 50
output_directory_path = os.path.join(input_path, 'topic-model', 'output', '50_topic')


path_to_training_data           = output_directory_path + '/training.txt'
path_to_formatted_training_data = output_directory_path + '/mallet.training'
path_to_model                   = output_directory_path + '/mallet.model.' + str(num_topics)
path_to_topic_keys              = output_directory_path + '/mallet.topic_keys.' + str(num_topics)
path_to_topic_distributions     = output_directory_path + '/mallet.topic_distributions.' + str(num_topics)
path_to_word_weights            = output_directory_path + '/mallet.word_weights.' + str(num_topics)
path_to_diagnostics             = output_directory_path + '/mallet.diagnostics.' + str(num_topics) + '.xml'



lmw.import_data(path_to_mallet,
                path_to_training_data,
                path_to_formatted_training_data,
                training_data)


lmw.train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      path_to_model,
                      path_to_topic_keys,
                      path_to_topic_distributions,
                      path_to_word_weights,
                      path_to_diagnostics,
                      num_topics)

topic_keys = lmw.load_topic_keys(output_directory_path + '/mallet.topic_keys.' + str(num_topics))

for i, t in enumerate(topic_keys):
    print(i, '\t', ' '.join(t[:20]))

topic_distributions = lmw.load_topic_distributions(output_directory_path + '/mallet.topic_distributions.' + str(num_topics))

len(topic_distributions), len(topic_distributions[0])

assert(len(topic_distributions) == len(training_data))

# might look at references to actors?
for p, d in lmw.get_top_docs(training_data, topic_distributions, topic_index=34, n=500):
    print(round(p, 4), d)
    print()

topic_word_probability_dict = lmw.load_topic_word_distributions(output_directory_path + '/mallet.word_weights.' + str(num_topics))

len(topic_word_probability_dict)


for _topic, _word_probability_dict in topic_word_probability_dict.items():
    print('Topic', _topic)
    for _word, _probability in sorted(_word_probability_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(round(_probability, 4), '\t', _word)
    print()

lmw.get_js_divergence_topics(9, 16, topic_word_probability_dict)


target_labels = [1.0, 2.0, 3.0, 4.0, 5.0, "nan"]

lmw.plot_categories_by_topics_heatmap(book_ratings,
                                      topic_distributions,
                                      topic_keys, 
                                      output_directory_path + '/categories_by_topics.pdf',
                                      target_labels=target_labels,
                                      dim=(12,6))


# Assess top documents without cleaning
topic_matrix = pd.DataFrame(topic_distributions)
topic_matrix.columns = topic_matrix.columns.astype(str)
reviews_df = reviews_df.reset_index()

doc_topic_matrix = pd.merge(reviews_df, topic_matrix, left_index=True, right_index=True)
doc_topic_matrix.to_csv(output_directory_path + '/doc_topic_matrix_20.csv')


selected_columns = doc_topic_matrix[["book_id","book_title", "review_id", "date", "rating", "text", "num_likes", "shelves", "34"]]
new_df = selected_columns.copy()
new_df.to_csv(output_directory_path + '/doc_topic_matrix_adaptation_topic.csv')
sorted_df = new_df.sort_values("34", ascending=False)
sorted_df.head(2000).to_csv(output_directory_path + '/doc_topic_adaptation_topic_truncated.csv')


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def get_text_length(string_of_text):
    tokens = tokenizer.tokenize(string_of_text)
    length = len(tokens)
    return length


doc_topic_matrix["review_length"] = doc_topic_matrix["text"].apply(get_text_length)

# plot review length
doc_topic_matrix["review_length"].hist(bins = 50)
doc_topic_matrix["review_length"].describe()

long_reviews_df = doc_topic_matrix[doc_topic_matrix["review_length"] > 100]
selected_columns_long = long_reviews_df[["book_id","book_title", "review_id", "date", "rating", "text", "num_likes", "shelves", "34"]]
new_long_df = selected_columns_long.copy()
sorted_long_df = new_long_df.sort_values("34", ascending=False)
sorted_long_df.head(2000).to_csv(output_directory_path + '/doc_topic_adaptation_topic_long.csv')