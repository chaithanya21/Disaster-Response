'''
ML Pipeline
'''

#import required libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import  WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    '''
    Loads the data from an Sql Database
    
    Parameters
    
    database_filepath : path to the database file
    
    -------
    Returns
    X : faetures which are actually the messages
    Y : class labels (different categories of classes)
    category_names : Names of all labels as list of string`
    -------
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query('SELECT ' + '*' + ' FROM ' + engine.table_names()[0] +';', engine)
    X =df.message
    Y=df.drop(columns=['message', 'original', 'genre', 'id'])
    category_names = Y.columns.tolist()
    
    return X.values,Y.values,category_names


def tokenize(text):
    '''
    tokenizes each message in the datframe
    using word_tokenizer and reduces each token into its root form using lemmatizer
    
    parameters
    text: actual text message from the features dataframe (A String)
    
    --------
    Returns
    clean_tokens:a lits of cleaned tokens
    
    --------
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #removes punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    
    return clean_tokens


def build_model(grid_search=False):
    '''
    build a Machine Learning pipeline using Countvectorizer,TfidfTransformer and MultiOutputClassifier
    
    parameters
    grid_search: a boolean parameter which is by default set as False
    if specified as True, Grid Search CV is used for hyperparameter tuning.
    
    -------
    
    Returns
    Pipeline:An sklearn pipeline
    
    -------
    '''
    #build the MAchine Learning pipeline using sklearn
    
    pipeline=Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1,
                                                       min_samples_split = 2,
                                                       bootstrap=True,
                                                       n_estimators=100,)))
    ])
    
    if grid_search:
        parameters={
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__bootstrap': [True],
        'clf__estimator__n_estimators': [100, 200, 300]
        }
        
        cv =GridSearchCV(pipeline,parameters,n_jobs = -1)
        return cv
    else:
        return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the model using classification report
    on all the 36 categories
    '''
    y_pred = model.predict(X_test)
    accuracy=(y_pred==Y_test).mean()
    print('The accuracy of the entire model is:',round(accuracy,3))
    
    for i in range(len(category_names)):
        accuracy=(y_pred[:,i]==Y_test[:,i]).mean()
        print('accuracy for the category {} is {}'.format(category_names[i], round(accuracy,3)))
        print(classification_report(Y_test[:,i], y_pred[:,i]))
        print('-'*60)
      
def save_model(model, model_filepath):
    '''
    saves the results of a Trained model as pkl file
    
    parameters
    model:Final trained model with best optimal solution
    model_filepath: path to save the model
    
    -------
    Returns
    
    None
    
    -------
    
    '''
    joblib.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(grid_search=False)
        #print(model.best_estimator_)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()