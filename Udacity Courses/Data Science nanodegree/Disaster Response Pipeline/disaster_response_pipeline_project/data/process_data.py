# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from CSV files and merge them into a single DataFrame.

    This function loads message data and category data from two separate CSV files, 
    and merges them into a single DataFrame based on the 'id' column.

    Args:
        messages_filepath (str): The file path of the messages CSV file.
        categories_filepath (str): The file path of the categories CSV file.

    Returns:
        df: The merged Pandas DataFrame containing both the message and category data.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Clean a DataFrame by splitting the 'categories' column into multiple columns and 
    converting values to binary (0 or 1).

    This function splits the 'categories' column of the DataFrame into multiple columns, 
    one for each category, and converts the values in these new columns to 0 or 1. The 
    original 'categories' column is dropped, and any duplicates in the DataFrame are removed.

    Args:
        df: The DataFrame to be cleaned. It is expected to contain a 
            'categories' column where values are semicolon-separated strings.

    Returns:
        df: The cleaned DataFrame. It will contain one new column for each unique 
            category in the original 'categories' column. The values in these new 
            columns will be 0 or 1. Any duplicates in the original DataFrame
            have been removed.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save DataFrame into a SQLite database.

    This function takes a DataFrame and a database filename, and stores the DataFrame 
    as a table in a SQLite database. If a table with the same name already exists in the 
    database, it is replaced.

    Args:
        df: The DataFrame to be saved.
        database_filename (str): The filename of the SQLite database.

    Returns:
        None
    """

    # Create a SQLAlchemy engine instance connected to the SQLite database
    engine = create_engine('sqlite:///'+ database_filename)

    # Write the DataFrame to the SQLite database, replacing any existing table with the same name
    df.to_sql('Messages_table_v1', engine, index=False, if_exists = 'replace')

    
def main():
    """
    Main function to load, clean, and save data for disaster response.

    This function runs the following process:
    - Load messages and categories data from CSV files.
    - Clean the merged data.
    - Save the cleaned data into a SQLite database.
    
    The file paths of the messages data, categories data, and the database should be 
    provided as command-line arguments in that order.

    Returns:
        None
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()