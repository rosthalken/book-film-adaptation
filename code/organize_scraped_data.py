import pandas as pd
import os
import numpy as np
import json
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt


from adaptation_functions import plot_by_review, plot_distribution

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


# Retrieve book to film metadata
book_to_film_file_name = 'metadata_book_film.csv'
book_to_film_path = os.path.join(input_path, 'data', book_to_film_file_name)
book_to_film_df = pd.read_csv(book_to_film_path)


len(book_to_film_df.book_info.unique())
len(book_to_film_df.movie_info.unique())

# remove columns in book_to_film_df that are NA --> results in 3147 rows
book_to_film_df = book_to_film_df.dropna(subset=['id'])

# then turn into int values using book_to_film_df["id"].astype(int)
book_to_film_df["id"] = book_to_film_df["id"].astype(int)


# Link to book metadata from Goodreads
file_name = 'all_books.csv'
path_to_book_meta = os.path.join(input_path, 'get_books_output', file_name)
book_meta_df = pd.read_csv(path_to_book_meta)
book_meta_df = book_meta_df.add_suffix("_book")

# Match by book_meta_df "book_id" to book_to_film["id"]
# Change columns so not mixed up

# left join book_meta_df["book_id"] to book_to_film_df["id"] and allow duplicate rows from book_to_film_df
book_meta_to_film_df = pd.merge(book_to_film_df, book_meta_df, left_on="id", right_on= "book_id_book")


### Now, link to imdb data
# https://github.com/dojutsu-user/IMDB-Scraper
# Cite this paper: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib

imdb_file_name = 'imdb_dataset.json'
imdb_path = os.path.join(input_path, 'data', imdb_file_name)

with open(imdb_path, 'r') as j:
     imdb_content = json.loads(j.read())

imdb_df = pd.DataFrame(imdb_content)
imdb_df = imdb_df.add_suffix("_film")
merged_df = pd.merge(book_meta_to_film_df, imdb_df, left_on=['film_title','film_year'], right_on = ['title_film','year_film'])

# reduce to no na for match:
connected_titles_df = merged_df[merged_df['title_film'].notna()]
len(connected_titles_df["book_title"].unique())
# 1060

len(connected_titles_df["title_film"].unique())
# 1207

connected_titles_df.to_csv(os.path.join(input_path, 'data', 'connected_titles.csv'))





book_year_df = book_to_film_df.dropna(subset=['book_year'])
book_year_df = book_year_df[book_year_df["book_year"] != "unknown"]
# then turn into int values using book_to_film_df["id"].astype(int)
book_year_df["book_year"] = book_year_df["book_year"].astype(float)
book_year_df = book_year_df[book_year_df["book_year"] != 2419.0]
book_df = pd.DataFrame()
book_df["year"] = book_year_df["book_year"]
book_df["media"] = "book"

film_year_df = book_to_film_df.dropna(subset=['film_year'])
film_year_df = film_year_df[film_year_df["film_year"] != "unknown"]
# then turn into int values using book_to_film_df["id"].astype(int)
film_year_df["film_year"] = film_year_df["film_year"].astype(float)
film_year_df = film_year_df[film_year_df["film_year"] != 3000.0]
film_year_df = film_year_df[film_year_df["film_year"] != 1001.0]
film_df = pd.DataFrame()
film_df["year"] = film_year_df["film_year"]
film_df["media"] = "film"


frames = [film_df, book_df]
year_df = pd.concat(frames)

year_df.to_csv(output_path + '/year_df.csv')

output_path = os.path.join(input_path, 'output')
figure_output = os.path.join(output_path, 'figures')


plot_distribution(year_df, "year", "media")

connected_titles_df["users_rating_film"]
connected_titles_df["users_rating_film"] = connected_titles_df["users_rating_film"].astype(float)
sns.histplot(connected_titles_df, x = "users_rating_film", bins = 20)


connected_titles_df["average_rating_book"]
sns.histplot(connected_titles_df, x = "average_rating_book", bins = 20)

# fix this issue
connected_titles_df["votes_film"] = connected_titles_df["votes_film"].astype(int)

connected_titles_df.plot.scatter(x="average_rating_book", y = "users_rating_film")


cols = ["users_rating_film", "average_rating_book", "votes_film", "num_ratings_book"]
for col in cols:
    col_zscore = col + '_zscore'
    connected_titles_df[col_zscore] = (connected_titles_df[col] - connected_titles_df[col].mean())/connected_titles_df[col].std(ddof=0)

plot_by_review(connected_titles_df, "users_rating_film_zscore", "average_rating_book_zscore")

connected_titles_df.to_csv(output_path + '/connected_metadata.csv')


# 
# def plot_by_review(data, col_1, col_2):
#     colors = ["#088da5", "#8da508", "#a5088d", "#ff6f61"]
#     palette ={"book": colors[2], "film": colors[0]}
#         
#     # box plot
# 
#     plt.figure(figsize=(10,7))
#     ax = sns.scatterplot(data = data, x = col_1, y = col_2)
#     plt.xlabel('Film Rating', fontsize = 15)
#     plt.ylabel('Book Rating', fontsize = 15)
#     #plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
#     plt.ylim([-5, 2.5])
#     #plt.xlim([2, 9])
#     #sns.despine()
# 
#     plt.savefig(figure_output + '/ratings', dpi = 300)
#     plt.show()



# 
# def plot_distribution(df, column_plot, column_color):
#     sns.set_palette(sns.color_palette("colorblind"))
#     colors = ["#088da5", "#8da508", "#a5088d", "#ff6f61"]
#     palette ={"book": colors[2], "film": colors[0]}
#         
#     # box plot
#     
#     plt.figure(figsize=(10,7))
#     sns.histplot(df, x=column_plot, bins = 100, hue = column_color, alpha = .5, palette=palette);
#     plt.xlabel('Year', fontsize = 15)
#     plt.ylabel('Count', fontsize = 15)
#     sns.despine()
# 
#     plt.savefig(figure_output + '/year_wider_dist', dpi = 300)
#     plt.show()

