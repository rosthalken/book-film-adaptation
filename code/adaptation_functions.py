import os
import pandas as pd
import numpy as np
import re
from collections import Counter


# From a csv with an id column, get unique ids and write them to a txt
# Use case: I have Goodreads IDs but they aren't unique, I don't want to scrape something twice

def get_unique_ids_to_txt(data_path, output_path):
    input_path = os.getcwd()
    csv_with_ids = pd.read_csv(os.path.join(input_path, 'data', data_path))
    ids = pd.DataFrame(csv_with_ids.id.unique())
    sorted_ids = ids.sort_values(by = 0).dropna().astype(int)
    pd.DataFrame.to_csv(sorted_ids, os.path.join(input_path, 'data', output_path), index = False)

#get_unique_ids_to_txt('metadata_book_film.csv', 'ids_test.txt')


def split_string_after_year(df_cell):
    delimiter = '\n'
    string_l = df_cell.split(delimiter)
    return string_l


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



def plot_by_review(data, col_1, col_2):
    colors = ["#088da5", "#8da508", "#a5088d", "#ff6f61"]
    palette ={"book": colors[2], "film": colors[0]}
        
    # box plot

    plt.figure(figsize=(10,7))
    ax = sns.scatterplot(data = data, x = col_1, y = col_2)
    plt.xlabel('Film Rating', fontsize = 15)
    plt.ylabel('Book Rating', fontsize = 15)
    plt.ylim([-5, 2.5])

    plt.savefig(figure_output + '/ratings', dpi = 300)
    plt.show()



def plot_distribution(df, column_plot, column_color):
    sns.set_palette(sns.color_palette("colorblind"))
    colors = ["#088da5", "#8da508", "#a5088d", "#ff6f61"]
    palette ={"book": colors[2], "film": colors[0]}
        
    # box plot
    
    plt.figure(figsize=(10,7))
    sns.histplot(df, x=column_plot, bins = 100, hue = column_color, alpha = .5, palette=palette);
    plt.xlabel('Year', fontsize = 15)
    plt.ylabel('Count', fontsize = 15)
    sns.despine()

    plt.savefig(figure_output + '/year_wider_dist', dpi = 300)
    plt.show()
