import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import pickle
from sqlalchemy import create_engine

import sys 
sys.path.append('..')

from models.train_classifier import TextPreprocesser, TextVectorizer

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)
categories = df.columns[4:]

# load model
model = pickle.load(open('../models/pretrained_classifiers/classifier.pkl', 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # categories
    category_split = (df.groupby(by=['genre'])[categories[1:]].sum()/df[df['related']==1]['related'].sum()*100).reset_index()
    category_split = pd.melt(category_split, value_vars=categories[1:], 
                            id_vars='genre', var_name='category', value_name='share (%)')
    cat_stat = ((df[categories[1:]].sum()/df[df['related']==1]['related'].sum()*100)
                .reset_index()
                .rename(columns={'index': 'category',
                                0: 'share total'})
            )
    category_split = category_split.merge(cat_stat, on='category', how='left')
    category_split = category_split.sort_values(by='share total', ascending=False)
   
    # create visuals
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
        },
        {
            'data': [
                Bar(
                    x=category_split[category_split['genre']=='direct']['category'],
                    y=category_split[category_split['genre']=='direct']['share (%)'],
                    name='direct'
                ),
                Bar(
                    x=category_split[category_split['genre']=='news']['category'],
                    y=category_split[category_split['genre']=='news']['share (%)'],
                    name='news'
                ),
                Bar(
                    x=category_split[category_split['genre']=='social']['category'],
                    y=category_split[category_split['genre']=='social']['share (%)'],
                    name='social'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Share (%)"
                },
                'xaxis': {
                    'title': "Category"
                },
                'barmode': 'stack'
            },
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()