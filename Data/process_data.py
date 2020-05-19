
'''
This file takes the dataset and follows an ETL(extract,transform,load) pipeline to create a cleaned dataset
which is then stored in a database
'''

#import the required files
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


#this function loads in the messages and categories dataset
def load_data(messages_filepath, categories_filepath):
    '''
    loads the datasets
    
    Parameters
    message_filepath: path to the messages csv file
    categories_filepath: path to the categories csv file
    
   
   -------------
    Returns merged pandas dataframe
    
    -------------
 
    '''
    
    message_df=pd.read_csv(messages_filepath)
    categories_df=pd.read_csv(categories_filepath)
    df =message_df.merge(categories_df,on='id')
    
    return df

#data cleaning
def clean_data(df):
    
    '''
    cleans the dataset and extracts features from the categories coloum
    The duplicate rows are also dropped
    
    parameters : df
    ----------
    Returns the cleaned dataset
    -----------
    
    '''
    categories=df['categories'].str.split(';',expand=True) #splits the categories columns based ';' and expands
    category_colnames =list() # a list which holds all the categories
    row =categories.iloc[0] #the first row of the categories df
    # use this row to extract a list of new column names for categories.
    for i in row:
        Find=i.find('-')
        category_colnames.append(i[:Find])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column]=categories[column].str.split('-').str.get(-1)
        # convert column from string to numeric
        categories[column] =categories[column].astype(int)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories],axis=1)
    # drop the original categories column from `df`
    df.drop(columns=['categories'],axis=1,inplace=True)
    # drop duplicates
    print('The Number of duplicate rows in the dataset:',df.duplicated(subset=None,keep='first').sum())
    df.drop_duplicates(subset=None,keep=False,inplace=True)
    # check number of duplicates after dropping
    print('The Number of duplicate rows in the dataset after dropping:',df.duplicated(subset=None,keep='first').sum())
        
    return df
        
#loads the cleaned dataframe to a database
def save_data(df, database_filename):
    
     '''
     Loads the df dataframe to a sql database
     
     Parameters
     df: dataframe
     database_filename : name of the database file
     ---------
     Returns
     ---------
     '''
     engine = create_engine('sqlite:///' + database_filename)
     df.to_sql('DisasterResponseTable', engine, index=False)
    
      
        


def main():
    
    '''
    Parameters
    _ _ _ _ _ _
    messages_filepath : path to the messages csv file
    categories_filepath : path to the categories csv file
    database_filepath : path where database file to be saved
 
    Returns
    _ _ _ _ _
    
    
    None
    
    
    '''
    
    
    
    
    
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
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
