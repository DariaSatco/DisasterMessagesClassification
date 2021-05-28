import sys
import pandas as pd
import logging
from sqlalchemy import create_engine

DEBUG_MODE = False
if DEBUG_MODE:
    log_level = logging.DEBUG
else:
    log_level = logging.INFO
    
logging.basicConfig(stream=sys.stdout, level=log_level)
logger_etl = logging.getLogger('ETL Pipeline')
logger_sql = logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load data from csv files and merge it into single pandas Dataframe

    Args:
        messages_filepath (str) : path to the file with messages data
        categories_filepath (str) : path to the file with categories data
    
    Return:
        DataFrame : table with merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id', how='left')
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess categories, extend one column into multiple columns
    with 1/0 values

    Args:
        df (DataFrame) : input table with categories columns

    Return:
        DataFrame : cleaned table
    """
    # split categories column on multiple columns
    splitted_categories = df['categories'].str.split(';', expand=True)
    category_colnames = [el.split('-')[0] for el in splitted_categories.loc[0]]
    splitted_categories.columns = category_colnames

    for column in splitted_categories:
        # set each value to be the last character of the string
        splitted_categories[column] = splitted_categories[column].apply(lambda x: x.split('-')[-1])
        
        # convert column from string to numeric
        splitted_categories[column] = splitted_categories[column].astype(int)
        # remove outliers (found that there are values = 2)
        splitted_categories[column] = splitted_categories[column].apply(lambda x: 1 if x>1 else x)

    # remove column where all values == 0
    all_zero_cols = list(splitted_categories.columns[splitted_categories.sum()==0])
    splitted_categories = splitted_categories.drop(columns=all_zero_cols)

    df = df.drop(columns=['categories'], errors='ignore')
    df = pd.concat([df, splitted_categories], axis=1).drop_duplicates(subset=['id'])

    return df


def save_data(df: pd.DataFrame, 
              database_filename: str) -> None:
    """
    Save DataFrame into SQL database

    Args:
        df (DataFrame) : table to save
    
    Return:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logger_etl.info('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        logger_etl.info('Cleaning data...')
        df = clean_data(df)
        
        logger_etl.info('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        logger_etl.info('Cleaned data saved to database!')
    
    else:
        logger_etl.error('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()