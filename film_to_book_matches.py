import os
import pandas as pd
import numpy as np
import re
from collections import Counter



file_path = os.path.join(os.getcwd(), 'book_to_film.csv')
output = os.path.join(os.getcwd(), 'output/metadata_book_film.csv')

book_to_film = pd.read_csv(file_path)


def split_string_after_year(df_cell):
    delimiter = '\n'
    string_l = df_cell.split(delimiter)
    return string_l

book_to_film = book_to_film[book_to_film['movie_info'].notnull()]

book_to_film['movie_info'] = book_to_film['movie_info'].apply(split_string_after_year)

book_to_film = book_to_film.explode('movie_info')

book_to_film = book_to_film.reset_index(drop=True)


def get_year(cell):
    try:
        year = re.findall(r'\d{4}', cell)[0]
    except IndexError:
        year = 'unknown'
    return year


def get_title(cell):
    try:
        year = re.findall(r'\d{4}', cell)[0]
        first_part = cell.split(year)[0]
        title = first_part.split(' (')[0]
    except IndexError:
        title = 'unknown'
    return title

def get_author(cell):
    try:
        year = re.findall(r'\d{4}', cell)[0]
        first_part = cell.split(year)[0]
        title = first_part.split(' (')[0]
        second_part = cell.split(year)[1]
        author = second_part.split('), ')[1]
    except IndexError:
        author = 'unknown'

    return author

def get_film_title(cell):
    try:
        year = re.findall(r'\d{4}', cell)[0]
        first_part = cell.split(year)[0]
        title = first_part.split(' (')[0]
    except IndexError:
        title = 'unknown'

    return title



# BOOK METADATA
book_to_film['book_year'] = book_to_film['book_info'].apply(get_year)
book_to_film['book_author'] = book_to_film['book_info'].apply(get_title)
book_to_film['book_title'] = book_to_film['book_info'].apply(get_author)

# FILM METADATA
book_to_film['film_year'] = book_to_film['movie_info'].apply(get_year)
book_to_film['film_title'] = book_to_film['movie_info'].apply(get_film_title)


book_to_film.to_csv(output)

