import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_table_v1', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Ignore first four columns ('id', 'message', 'original', 'genre')
    category_data = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Extract category names
    category_names = category_data.columns

    # Count non-zero entries for each category
    category_counts = category_data.apply(lambda x: (x != 0).sum())
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
#     graphs = [
#         {
#             'data': [
#                 Bar(
#                     x=genre_names,
#                     y=genre_counts
#                 )
#             ],

#             'layout': {
#                 'title': 'Distribution of Message Genres',
#                 'yaxis': {
#                     'title': "Count"
#                 },
#                 'xaxis': {
#                     'title': "Genre"
#                 }
#             }
#         },
#                     'data': [
#                 Bar(
#                     x=category_names,
#                     y=category_counts
#                 )
#             ],

#             'layout': {
#                 'title': 'Distribution of Message Categories',
#                 'yaxis': {
#                     'title': "Count"
#                 },
#                 'xaxis': {
#                     'title': "Category"
#                 }
#             }
#         }
#     ]
    
    def create_bar_graph(x_data, y_data, title, x_title, y_title="Count"):
        """
        Create a configuration for a bar graph suitable for Plotly.

        Parameters:
        - x_data (list): Data for the x-axis.
        - y_data (list): Data for the y-axis.
        - title (str): Title of the graph.
        - x_title (str): Title of the x-axis.
        - y_title (str, optional): Title of the y-axis. Defaults to "Count".

        Returns:
        - dict: A dictionary containing the configuration for the bar graph.
        """

        return {
            'data': [Bar(x=x_data, y=y_data)],
            'layout': {
                'title': title,
                'yaxis': {'title': y_title},
                'xaxis': {'title': x_title}
            }
        }

    graphs = [
        create_bar_graph(genre_names, genre_counts, 'Distribution of Message Genres', 'Genre'),
        create_bar_graph(category_names, category_counts, 'Distribution of Message Categories', 'Category')
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()