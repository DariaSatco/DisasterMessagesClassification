import sys

import pandas as pd
import numpy as np
from typing import List
from sqlalchemy import create_engine
import logging
from typing import List
import pickle
import os
import fnmatch

# gensim
from gensim.models import KeyedVectors

# NLTK
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, precision_score, recall_score


# download nltk packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

DEBUG_MODE = False
if DEBUG_MODE:
    log_level = logging.DEBUG
else:
    log_level = logging.INFO
    
logging.basicConfig(stream=sys.stdout, level=log_level)
logger_ml = logging.getLogger('ML Pipeline')
logger_sql = logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


def load_data(database_filepath: str):
    """
    Load X and Y from database

    Args:
        database_filepath (str) : path to the database with data

    Return:
        X (np.array) : X data
        Y (np.array) : Y data
        categories (List) : labels
    """

    engine = create_engine(f'sqlite:///{database_filepath}', echo=False)
    df = pd.read_sql_table('Messages', engine)
    categories = df.columns[4:]

    X = df['message'].values
    Y = df[df.columns[4:]].values

    return X, Y, categories


def get_wordnet_pos(word : str) -> str:
    """
    Map POS tag to the character that 
    nltk.stem.wordnet.WordNetLemmatizer().lemmatize() accepts
    
    Args:
        word (str) : the word for which you need to get the pos tag
    
    
    Return:
        str : pos tag, which is
            'n' for nouns and other unknowns
            'v' for verbs
            'a' for adjectives
            'r' for adverbs
    """
    
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']
    adjs = ['JJ', 'JJR', 'JJS']
    advs = ['RB', 'RBR', 'RBS']

    tag_dict = {wordnet.ADJ : adjs,
                wordnet.VERB : verbs,
                wordnet.ADV : advs}
    
    tag = nltk.pos_tag([word])[0][1]
    
    wordnet_pos = wordnet.NOUN
    for key in tag_dict:
        if tag in tag_dict[key]:
            wordnet_pos = key
    return wordnet_pos


def lemmatize(word : str) -> str:
    """
    Return lemma of the given word
    
    Args:
        word (str) : the word for which you need to get the lemma
    
    Return:
        str : lemma of the word
    
    """
    lemmed = WordNetLemmatizer().lemmatize(word, get_wordnet_pos(word))
    return lemmed


def tokenize(text : str) -> List:
    """
    Return tokens of the given text, which
    also pass through the following steps of 
    preprocessing:
        1) all words are put into lowercase 
        2) punctuation is removed
        3) stopwords are excluded
        4) each word is lemmatized
    
    Args:
        text (str) : any text in string format
    
    Return:
        List : list of filtered and preprocessed tokens 
    
    """
    tokens = word_tokenize(text)
    
    # lowercase and remove punktuation
    tokens = [w.lower() for w in tokens if w.isalpha()]
    
    # remove stopwords
    eng_stopwords = stopwords.words('english')
    filtered = [w for w in tokens if w not in eng_stopwords]
    
    # lemmatize
    lemmed = [lemmatize(w) for w in filtered]
    
    # if get an empty list after all filterings
    # return default list with single element ['null']
    if len(lemmed)==0:
        lemmed = ['null']
    
    return lemmed


class TextVectorizer():
    """
    This class is used to get a vector representation of a text
    based on word embeddings
    
    """
    
    def __init__(self, 
                 path=None):
    
        # Load pretrained model
        if not path:
            dirpath = 'models/pretrained_nlp_models/'
            path = dirpath + fnmatch.filter(os.listdir(dirpath), '*.bin')[0]
        logger_ml.debug('Load pretrained embedding model ...')
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)
        logger_ml.debug('The model was successfully loaded!')
        logger_ml.debug(f'The embeding size is {self.model.vector_size}')

    def get_word_vector(self, word : str) -> np.array:
        """
        Return word embedding
        
        Args:
            word (str)
        
        Return:
            np.array : word embedding
        """
        if word in self.model:
            return self.model[word]
        else:
            return self.model['null']
        

    def get_text_vector(self, text : str) -> np.array:
        """
        Return text embedding
        
        Args:
            text (str)
        
        Return:
            np.array : word embedding
        """
        tokens = tokenize(text)
        token_vecs = [self.get_word_vector(w) for w in tokens]
        text_vec = np.mean(token_vecs, axis=0)
        return text_vec


class TextPreprocesser(BaseEstimator, TransformerMixin):
    """
    Transform an array with texts into an array with text embedding vectors 
    """
    def __init__(self):
        self.text_vectorizer = TextVectorizer()
    
    def fit(self, X, y=None):
        return self

    def transform(self, X : np.array) -> np.array:
        """
        Args:
            X (np.array) : an array with texts of size (N,), 
                where N is the number of samples
        
        Return:
            np.array : an array of size (N, M), where N is the 
                number of samples and M is the embedding dimensionality
        """
        transformed = []
        for text in X:
            transformed.append(self.text_vectorizer.get_text_vector(text))
        transformed = np.array(transformed)
        return transformed


def build_model():
    """
    Build pipeline for further fitting
    """
    preprocesser = TextPreprocesser()  # text embeddings
    normalizer = Normalizer()
    clf = MultiOutputClassifier(LogisticRegression(), n_jobs=-1)

    pipeline = Pipeline([('preprocesser', preprocesser), 
                         ('normalizer', normalizer),
                         ('clf', clf)])
    return pipeline


def build_classification_report(Y_true: np.array, 
                                Y_pred: np.array, 
                                category_names: List) -> pd.DataFrame:
    """
    Build table with F1-score, precision and recall by category

    Args:
        Y_true (np.array) : test labels
        Y_pred (np.array) : predicted labels
        category_names (List) : categories
    Return:
        DataFrame
    """
    classification_report = pd.DataFrame(columns=['column', 'precision', 'recall', 'f1_score'])
    for i, col in enumerate(category_names):
        f1 = f1_score(Y_true[:,i], Y_pred[:,i], average='micro')
        precision = precision_score(Y_true[:,i], Y_pred[:,i], average='micro')
        recall = recall_score(Y_true[:,i], Y_pred[:,i], average='micro')
        classification_report = classification_report.append({'column': col,
                                                              'precision': precision,
                                                              'recall': recall,
                                                              'f1_score': f1}, 
                                                              ignore_index=True)
    return classification_report


def evaluate_model(model, 
                    X_test: np.array, 
                    Y_test: np.array, 
                    category_names: List) -> None:
    """
    Print scores of the model by each category separately and overall

    Args:
        model :
        X_test (np.array) : X-data
        Y_test (np.array) : Y-data
        category_names (List): categories

    Return:
        None
    """
    Y_preds = model.predict(X_test)

    clf_report = build_classification_report(Y_test, Y_preds, category_names)
    print(clf_report.round(3))
    
    f1 = f1_score(Y_test, Y_preds, average='micro')
    precision = precision_score(Y_test, Y_preds, average='micro')
    recall = recall_score(Y_test, Y_preds, average='micro')
    
    print(f'Micro-averaged F1-score {f1:4.2}')
    print(f'Micro-averaged precision {precision:4.2}')
    print(f'Micro-averaged recall {recall:4.2}')


def save_model(model, model_filepath: str) -> None:
    """
    Save trained model to file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logger_ml.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        logger_ml.info('Building model...')
        model = build_model()
        
        logger_ml.info('Training model...')
        model.fit(X_train, Y_train)
        
        logger_ml.info('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        logger_ml.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logger_ml.info('Trained model saved!')

    else:
        logger_ml.error('Please provide the filepath of the disaster messages database '\
                        'as the first argument and the filepath of the pickle file to '\
                        'save the model to as the second argument. \n\nExample: python '\
                        'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()