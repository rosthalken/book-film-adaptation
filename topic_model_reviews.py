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
# NOTE: WILL NEED TO reduce to only book reviews found in imdb

input_path = os.getcwd()
file_name = 'all_reviews.csv'
path_to_reviews = os.path.join(input_path, 'output', file_name)

reviews_df = pd.read_csv(path_to_reviews)

path_to_mallet = "/Users/rosamondthalken/Applications/Mallet-Dev/bin/mallet"

reviews_df["text"] = reviews_df["text"].astype(str)


# Using Maria Antoniak's little mallet wrapper (plus code from the lmw demo!)
training_data = [lmw.process_string(t) for t in reviews_df['text'].tolist()]
training_data = [d for d in training_data if d.strip()]

len(training_data)

book_ratings = reviews_df["rating"].to_list()
len(book_ratings)

lmw.print_dataset_stats(training_data)


### TOPIC MODEL ###
num_topics = 20
output_directory_path = os.path.join(input_path, 'topic_model', 'output')


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
    print(i, '\t', ' '.join(t[:10]))