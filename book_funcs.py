import math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def category_cleaner(df):
    df['Category'] = df['Category'].str.strip(r'[]\'.,\"')
    df['Category'] = df['Category'].str.lower()
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:11] != 'Young Adult' else 'Young Adult')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:11] != 'Young adult' else 'Young Adult')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:3] != 'Zoo' else 'Zoo')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:7] != 'Cookery' else 'Cookery')
    df['Category'] = df['Category'].apply(lambda txt: 'literary' if 'literary' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'biography & autobiography' if 'biography' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'biography & autobiography' if 'autobiography' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'history' if 'history' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'business & economics' if 'business' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'business & economics' if 'economics' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'psychology' if 'psychology' in txt else txt)
    df['Category'] = df['Category'].apply(
        lambda txt: 'fiction' if ('fiction' in txt) and ('juvenile' not in txt) and ('nonfiction' not in txt) else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'health & fitness' if 'health' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'philosophy' if 'philosophy' in txt else txt)
    big_cat = list(
        df.groupby(['Category'])['Category'].count()[df.groupby(['Category'])['Category'].count() > 90].sort_values(
            ascending=False).index)
    big_cat = big_cat[3:]
    for cat in big_cat:
        df['Category'] = df['Category'].apply(lambda txt: cat if cat in txt else txt)
        cat_lst = cat.split()
        df['Category'] = df['Category'].apply(lambda txt: cat if cat_lst[0] in txt else txt)
    return df


def df_cleaner(df, min_book_ratings: int = 5, min_user_ratings: int = 2):
    filter_books = df['isbn'].value_counts() > min_book_ratings
    filter_books = filter_books[filter_books].index.tolist()

    filter_users = df['user_id'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()
    print('The original data frame shape:\t{}'.format(df.shape))

    df_new = df[(df['isbn'].isin(filter_books))]  # &
    print('The data frame shape after bookk filtering:\t{}'.format(df_new.shape))
    df_new = df_new[df_new['user_id'].isin(filter_users)]
    print('The new data frame shape:\t{}'.format(df_new.shape))
    return df_new


def category_compliter(df):
    df_books = df.groupby('isbn').agg(
        {'book_title': 'first', 'book_author': 'first', 'year_of_publication': 'first', 'user_id': 'count',
         'age': 'mean', 'rating': 'mean', 'publisher': 'first', 'Category': 'first', 'img_s': 'first', 'img_m': 'first',
         'img_l': 'first', 'Summary': 'first',
         'Language': 'first', 'city': pd.Series.mode, 'state': pd.Series.mode, 'country': pd.Series.mode})

    df_author = df_books.groupby('book_author').agg(
        {'Category': pd.Series.mode, 'user_id': 'sum', 'publisher': 'count'})
    df_author['freq_Category'] = df_books.groupby('book_author')['Category'].agg(
        lambda x: x.mode()[0] if (x.mode()[0] != '9') or (len(x.mode()) == 1) else x.mode()[1])
    df_author['num_topic'] = df_author['Category'].apply(lambda lst: 1 if type(lst) == str else len(lst))
    df_author['topic9'] = df_author['Category'].apply(lambda lst: 1 if '9' in lst else 0)

    df_author = df_author[df_author['topic9'] == 1]
    df_author_relevant = df_author[df_author['num_topic'] > 1]
    df_author9_relevant_two_options = df_author_relevant[df_author_relevant['num_topic'] == 2]
    df_author9_relevant_two_options['pred_Category'] = df_author9_relevant_two_options['Category'].apply(
        lambda lst: lst[-1])

    df_add_cat = df_author9_relevant_two_options['freq_Category']
    add_lst = list(df_add_cat.index)
    df = df.reset_index()
    for i in range(len(df)):
        if df.loc[i, 'book_author'] in add_lst:
            df.loc[i, 'Category'] = df_add_cat[df.loc[i, 'book_author']]
    df = df.drop(['index', 'Unnamed: 0'], axis=1)
    return df

def zero_droper(df):
    df = df[df['rating']>0]
    df.reset_index(inplace = True, drop = True)
    return df