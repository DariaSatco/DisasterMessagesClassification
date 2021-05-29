import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Scatter
import pickle
from sqlalchemy import create_engine
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer

import sys 
sys.path.append('..')

from models.train_classifier import TextPreprocesser, TextVectorizer, load_data

app = Flask(__name__)

# load data
database_filepath = 'DisasterResponse.db'
engine = create_engine(f'sqlite:///../data/{database_filepath}')
df = pd.read_sql_table('Messages', engine)
X, Y, category_names = load_data(f'../data/{database_filepath}')

# load model
model = pickle.load(open('../models/pretrained_classifiers/classifier.pkl', 'rb'))

# features
prep_X_array = pd.read_sql_table('Embeddings', engine)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # categories
    category_split = (df.groupby(by=['genre'])[category_names[1:]].sum()/df[df['related']==1]['related'].sum()*100).reset_index()
    category_split = pd.melt(category_split, value_vars=category_names[1:], 
                            id_vars='genre', var_name='category', value_name='share (%)')
    cat_stat = ((df[category_names[1:]].sum()/df[df['related']==1]['related'].sum()*100)
                .reset_index()
                .rename(columns={'index': 'category',
                                0: 'share total'})
            )
    category_split = category_split.merge(cat_stat, on='category', how='left')
    category_split = category_split.sort_values(by='share total', ascending=False)

    # message embedding
    # prepare sample for visualization
    sample = pd.DataFrame(prep_X_array).sample(1000, random_state=0).copy()
    sample_index = sample.index

    frequent_cat = cat_stat[cat_stat['share total']>10]['category'].to_list()

    # build 2D projection
    X_embedded = TSNE(n_components=2).fit_transform(sample)
    proj_df = pd.concat([pd.DataFrame(X_embedded, columns=['x1', 'x2']), 
                     df.loc[sample_index,['message']+frequent_cat].reset_index(drop=True)], 
                    axis=1)
    proj_df = pd.melt(proj_df, value_vars=frequent_cat, id_vars=['x1', 'x2', 'message'], 
                    value_name='present', var_name='category')

    button_labels = []
    figure_list = []
    for i, cat in enumerate(frequent_cat):
        selection = proj_df[proj_df['category']==cat].copy()
        if i==0:
            figure_list.append(Scatter(x=selection['x1'], 
                                y=selection['x2'],  
                                hoverinfo='text',
                                text=selection['message'].to_list(),
                                name=cat,
                                mode='markers', 
                                marker=dict(color=selection['present'],
                                            opacity=0.3, 
                                            colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                                            size=10),
                                )
                            )
        else:
            figure_list.append(Scatter(x=selection['x1'], 
                                y=selection['x2'],  
                                hoverinfo='skip',
                                name=cat,
                                mode='markers', 
                                marker=dict(color=selection['present'],
                                            opacity=0.3, 
                                            colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                                            size=10),
                                )
                            )


        button = dict(label=cat,
                    method="restyle", 
                    args=[{'x': [selection['x1']],
                            'y': [selection['x2']],
                            'marker': dict(color=selection['present'],
                                        opacity=0.2, 
                                        colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
                                        size=10)}],
                    )
        button_labels.append(button)
    
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
        },
        {
        'data' : figure_list,
        'layout' : dict(
            title = 'Message representation in 2D space for the most frequent categories',
            showlegend=False,
            width=1000,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    active=0,
                    showactive=True,
                    buttons=button_labels,
                    ),
                ]
        )
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