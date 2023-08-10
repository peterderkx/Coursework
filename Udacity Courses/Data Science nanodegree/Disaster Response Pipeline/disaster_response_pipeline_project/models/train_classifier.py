# Standard library imports
import sys
import re
import pickle
import warnings

# Third-party imports
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



def load_data(database_filepath):
    """
    Load data from a SQLite database and return feature and target variables.
    
    Args:
    -----
    - database_filepath (str): Path to the SQLite database file.
    
    Returns:
    --------
    - X (pd.Series): Messages as the feature.
    - Y (pd.DataFrame): Categories as the target variables.
    - category_names (pd.Index): Column names of Y.
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages_table_v1', engine)

    # Define feature and target variables X and Y
    # 'message' is my feature
    X = df['message']  

    # ignore columns 'id','message','original','genre' so all other columns thereafter are my target
    Y = df.iloc[:, 4:] 

    # define category names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text data

    Args:
    -----------
    text str: Messages as text data

    Returns:
    -----------
    clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    
    # Lemmatize verbs by specifying pos
    clean_tokens = [lemmatizer.lemmatize(w, pos='v') for w in lemmatized]
    
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and initialize grid search on the pipeline.
    
    This function constructs a pipeline that:
    1. Vectorizes the input text data using CountVectorizer with a custom tokenizer.
    2. Transforms the vectorized data using TfidfTransformer.
    3. Uses a MultiOutputClassifier with a RandomForestClassifier as the estimator for classification.
    
    Grid search is then initialized on this pipeline to find the best parameters from a given set.
    
    Returns:
    --------
    cv: GridSearchCV object
        Initialized grid search object on the pipeline.
    """
    # Suppress all warnings
    warnings.simplefilter("ignore")

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # I did this training already in NL pipeline Preparation
    # Training can take quite a while, hence I provide the code here but will run with optimized settings

    # Define parameters for GridSearch
    # Regularization by increase minimum number of samples required at a leaf node to prevent overfitting
    parameters = {
        'clf__estimator__min_samples_leaf': [1] 
        # 'clf__estimator__min_samples_leaf': [1, 2] were tested    
    }

    # Initialize GridSearch, limit cross validation to 3 fold to speed up gridsearch
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=1)
    # cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3) 3 fold cross validation was tested

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning model's performance on test data.
    
    This function takes in a machine learning model, test data, and category names to 
    evaluate the model's performance using precision, recall, and F1-score metrics.
    The evaluation results are then printed in a tabular format for each category.

    Args:
    -----
    model : estimator object
        The machine learning model to be evaluated.

    X_test : pd.Series or array-like
        The test feature data.
        
    Y_test : pd.DataFrame or array-like
        The actual labels for the test set.

    category_names : list
        List of strings indicating the names of the categories.
        
    Returns:
    --------
    None. This function only prints the evaluation results.
    """
    
    # Predict on the test set
    Y_pred = model.predict(X_test)

    # Collecting the metrics
    metrics_list = []

    # Calculate and store the metrics for each output category
    for i, column in enumerate(category_names):
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test.iloc[:, i], Y_pred[:, i], average='weighted', zero_division=1)
        metrics_list.append([column, precision, recall, fscore])

    # Convert metrics list to a DataFrame
    metrics_df = pd.DataFrame(metrics_list, columns=['Category', 'Precision', 'Recall', 'F1-score'])

    # print evaluation metrics
    print(metrics_df)



def save_model(model, model_filepath):
    """
    Save the trained model to a file using Python's pickle module.
    
    This function takes in a machine learning model and a file path, 
    and then saves the model to the specified file using pickle. This 
    allows the model to be loaded and used later without needing to be retrained.

    Args:
    -----------
    model : estimator object
        The trained machine learning model to be saved.
        
    model_filepath : str
        The path where the model should be saved. This includes the name of the file 
        and its extension (which should typically be ".pkl" for pickle files).
        
    Returns:
    --------
    None. The function saves the model to the specified path and doesn't return anything.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    """
    Main function to execute the sequence of operations for training a classifier.
    
    This function does the following:
    1. Loads data from a specified database filepath.
    2. Splits the data into training and testing sets.
    3. Builds a machine learning model.
    4. Trains the model using the training data.
    5. Evaluates the model using the testing data.
    6. Saves the trained model to a specified model filepath.
    
    Command Line Arguments:
        1. Path to the SQLite database containing pre-processed data.
        2. Path to save the trained model as a pickle file.
    
    Usage:
        python train_classifier.py [database_filepath] [model_filepath]
    
    Example:
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    
    Returns:
    --------    
        None. Prints out the status of each step (loading, training, evaluating, saving) and saves the trained model.
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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