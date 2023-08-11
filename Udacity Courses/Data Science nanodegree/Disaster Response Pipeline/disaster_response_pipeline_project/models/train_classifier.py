# Standard library imports
import sys
import re
import pickle

# Libraries for data handling and storage
import pandas as pd
from sqlalchemy import create_engine
import pickle

# Text processing libraries
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Machine learning libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Others
import warnings


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
    3. Uses a MultiOutputClassifier with a AdaBoostClassifier as the estimator for classification.
    
    Grid search is then initialized on this pipeline to find the best parameters from a given set.
    
    Returns:
    --------
    cv: GridSearchCV object
        Initialized grid search object on the pipeline.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Define parameters for GridSearch, with other features besides tfidf:
    parameters = {
        'clf__estimator__learning_rate': [0.2, 0.5],
        'clf__estimator__n_estimators': [10, 20, 30]
      }

    # Initialize GridSearch, limit cross validation to 3 fold
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning model's performance on test data.
    
    The report contains precision, recall, and F1-score for each class (0 and 1) 
    and their weighted average for each category in the test dataset.

    Parameters:
    - Y_test (pd.DataFrame): The true labels for each category.
    - Y_pred (np.ndarray): Predicted labels from the model.
    - model: The machine learning pipeline used for predictions. 
    
    Returns:
    --------    
    - df_reports: A dataframe containing precision, recall, and F1-score for each 
                  class and their weighted averages for each category.

    Notes:
    - It makes use of the `classification_report` from sklearn.metrics.
    - The dataframe will have columns for each metric for classes '0' and '1' and 
      their weighted averages.
    - Results are rounded to two decimal places.
    """
    
    # Predict on test data:
    Y_pred = model.predict(X_test)

    # List to store the results for each category
    reports = []

    # Loop through each category
    for i, col in enumerate(Y_test.columns):
        
        # Get the classification report for each column
        report = classification_report(Y_test[col], Y_pred[:, i], zero_division=1, output_dict=True)
        
        # Extracting the metrics of interest
        precision_0 = report['0']['precision'] if '0' in report else None
        recall_0 = report['0']['recall'] if '0' in report else None
        f1_0 = report['0']['f1-score'] if '0' in report else None

        precision_1 = report['1']['precision'] if '1' in report else None
        recall_1 = report['1']['recall'] if '1' in report else None
        f1_1 = report['1']['f1-score'] if '1' in report else None

        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1 = report['weighted avg']['f1-score']

        # Store the results in a dictionary
        reports.append({
            'Category': col,
            'Precision_0': precision_0,
            'Recall_0': recall_0,
            'F1_0': f1_0,
            'Precision_1': precision_1,
            'Recall_1': recall_1,
            'F1_1': f1_1,
            'Weighted_Avg_Precision': weighted_avg_precision,
            'Weighted_Avg_Recall': weighted_avg_recall,
            'Weighted_Avg_F1': weighted_avg_f1
        })

    # Convert the results into a DataFrame
    df_reports = pd.DataFrame(reports)
    
    # Round numbers to 2 digits 
    df_reports = df_reports.round(2)

    # Display results
    print(df_reports)


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
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    
    Returns:
    --------    
        None. Prints out the status of each step (loading, training, evaluating, saving) and saves the trained model.
    """

    # Suppress all warnings
    warnings.simplefilter("ignore")

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