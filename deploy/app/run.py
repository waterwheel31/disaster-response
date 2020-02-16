import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle

import sys
sys.path.append('../models')

import train_classifier


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
engine = create_engine('sqlite:///../data/database.db')
df = pd.read_sql_table('data', engine)

# load model
#model = joblib.load('../models/finalized_model.pkl')
model_path = '../models/finalized_model.pkl'
model = pickle.load(open(model_path, 'rb'))
vectorizer_path = '../models/vectorizer.pkl'
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
tfidf_path = '../models/tfidf.pkl'
tfidf = pickle.load(open(tfidf_path, 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
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
    print('click')
    query = request.args.get('query', '') 

    # use model to predict classification for query
    num_col = 36
    error_cols = [9]

    predict = train_classifier.predict(model, [query], num_col, vectorizer, tfidf, error_cols)
    print('predict:', predict)

    print('classification_labels:', classification_labels)

    print('df:', df.columns[4:])

    #classification_results = dict(zip(df.columns[4:][0], classification_labels))
    classification_results = {df.columns[4:][0] : classification_labels}

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()