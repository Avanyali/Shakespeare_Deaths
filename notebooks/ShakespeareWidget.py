import pandas as pd
import numpy as np

import shakespeare_functions as sf
import shakespeare_dicts as sd

import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from math import exp
import re

import spacy
import en_core_web_sm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk import RegexpTokenizer

import streamlit as st


# Import allowable columns.

line_head = pd.read_csv('../data/csv/ShakespeareCharacterLines_engineered.csv', index_col = ['play', 'name', 'line_number'], \
                     nrows = 0)

original_char_df = pd.read_csv('../data/csv/ShakespeareCharacterLines_character_corpus.csv', index_col = ['play', 'name'])
original_shape = original_char_df.shape


columns = line_head.columns[9:].tolist()

input_lines = pd.DataFrame(columns = ['input_name', 'line_number', 'character_line'])
input_lines.set_index(['input_name', 'line_number'], inplace = True)

colon = slice(None)

# Import models
type_logreg = pickle.load(open('../models/TypeLogisticRegression.pkl', 'rb'))

char_svc = pickle.load(open('../models/CharPCASVC.pkl', 'rb'))

input_sen = st.text_area('Put your text here:')

curr = 'Character'

input_sens = input_sen.split('\n')
for i in range(0, len(input_sens)):
    input_lines.loc[(curr, i), 'character_line'] = input_sens[i]

def fix_tabs_newlines(string):
    #Turn multiple whitespace between two characters to a single whitespace.
    out = re.sub('(?<=\S)\s+(?=\S)', ' ', string)
    #Remove preceding whitespace
    out = re.sub('(?<=\A)\s+', '', out)
    #Remove ending whitespace
    out = re.sub('\s+(?=\Z)', '', out)
    #Remove preceding ',' plus whitespace.
    out = re.sub('\A,\s+(?=\S)', ' ', out)
    return out

input_lines['character_line'] = input_lines['character_line'].map(fix_tabs_newlines)

# Remove excess punctuation.
input_lines['character_line'] = input_lines['character_line'].map(sf.clean_punctuation)

# Separate all contractions, replace all poetic apostrophes with correct lettering.
input_lines['character_line'] = input_lines['character_line'].map(sf.clean_contractions)

# Remove ``’s''s, they will be removed by lemmatizing or stopwords anyway.
input_lines['character_line'] = input_lines['character_line'].map(lambda x: re.sub("\'s", '', x))

# Modernize anachronistic/poetic words.
input_lines['character_line'] = input_lines['character_line'].map(sf.clean_anachronisms)

input_lines['character_line'] = input_lines['character_line'].map(lambda x: re.sub(r"-", ' ', x))

# Bring in spaCy for lemmatization, stopwords, NER, and POS tagging.
nlp = en_core_web_sm.load()

# Add personalized stopwords.
for stop in sf.get_stopwords():
    nlp.vocab[stop].is_stop = True

# Get sentiment analysis of lines without proper nouns.
input_lines['pos_sentiment'] = 0
input_lines['neg_sentiment'] = 0
input_lines['neu_sentiment'] = 0
input_lines['compound_sentiment'] = 0

sia = SentimentIntensityAnalyzer()

for name, line_number, character_line, pos_sentiment, neg_sentiment, \
    neu_sentiment, compound_sentiment, in input_lines.to_records():
    
    line = character_line
    
    doc = nlp(line)
    to_sent = ""
    
    for token in doc:
        if token.pos_ != 'PROPN':
            to_sent += token.text + " "
       
    sent = sia.polarity_scores(to_sent)
    
    for key, value in sent.items():
        if input_lines[f'{key}_sentiment'].dtypes != 'float32':
            input_lines[f'{key}_sentiment'] = input_lines[f'{key}_sentiment'].astype(dtype='float32')
        input_lines.loc[(name, line_number), f'{key}_sentiment'] = value

input_lines['compound_sentiment'] = input_lines['compound_sentiment'].map(lambda x: (x + 1)/2)

# Final tokenization: remove stopwords, add columns for words, hypernyms, and word types
def dict_increment(a_dict, word):
    if word in a_dict:
        a_dict[word] += 1
    else:
        a_dict[word] = 1

character_names = set(input_lines.index.get_level_values(1))

for name, line_number, character_line, pos_sentiment, \
    neg_sentiment, neu_sentiment, compound_sentiment in input_lines.to_records():
    
    line = character_line
    
    doc = nlp(line)
    block = {}
    
    for token in doc:
        features = []
        
        if token.text.upper() in character_names:
            features.extend(["character_name", "PROPN"])
        else:
            if not token.is_stop:
                features.extend([token.lemma_.lower(), token.pos_])

                syn_text = wn.synsets(token.text.lower())
                syn_lemma = wn.synsets(token.lemma_.lower())

                #Add hypernym/synonym columns in priority order: word hypernym > lemma hypernym > word synonym > lemma synonym.
                if len(syn_text) > 0 and len(syn_text[0].hypernyms()) > 0:
                    dict_increment(block, syn_text[0].hypernyms()[0].lemma_names()[0].lower() + "_hyp")
                elif len(syn_lemma) > 0 and len(syn_lemma[0].hypernyms()) > 0:
                    dict_increment(block, syn_lemma[0].hypernyms()[0].lemma_names()[0].lower() + "_hyp")
                elif len(syn_text) > 0:
                    dict_increment(block, syn_text[0].lemma_names()[0].lower() + "_syn")
                elif len(syn_lemma) > 0:
                    dict_increment(block, syn_lemma[0].lemma_names()[0].lower() + "_syn")
                
        for item in features:
            dict_increment(block, item)

    for word, count in block.items():
        if not word in input_lines.columns:
            input_lines[word] = 0
            input_lines[word] = input_lines[word].astype(dtype='uint16')
        input_lines.loc[(name, line_number), word] = count


input_lines.fillna(value = 0, inplace = True)


# Wrangle frame into correct shape.

input_lines.drop(columns= ['character_line'], inplace = True)

for col in input_lines.columns.tolist():
    if col not in columns:
        input_lines.drop(columns= [col], inplace = True)

input_cols = input_lines.columns.tolist()
for col in columns:
    if col not in input_cols:
        input_lines[col] = 0

# Create aggregated dataframe by character for modeling.
index_without_number = set(input_lines.index.get_level_values(0))

for name in index_without_number:
    character_slice = input_lines.loc[(name, colon)]

    means = character_slice.mean().add_suffix('_mean')
    medians =   character_slice.median().add_suffix('_median')
    stds = character_slice.std(ddof = 0).add_suffix('_std')
    
    new_row = pd.concat([means, medians, stds], axis = 0)
    new_row.name = ('widget', name)

    original_char_df = original_char_df.append(new_row)

# Recreate PCA for models.
original_char_df.drop(columns = 'character_dies', inplace = True)

original_char_df.fillna(0, inplace = True)

char_xlist = original_char_df.columns.tolist()

char_X = original_char_df[char_xlist]

sc = StandardScaler()
char_Xs = sc.fit_transform(char_X)

pca_sm = PCA(random_state=42, n_components=1114)
char_Zs = pca_sm.fit_transform(char_Xs)

preds = char_svc.predict_proba(char_Zs)[original_shape[0]:]

names = original_char_df.loc[('widget', colon)].index

predictions = zip(names, preds)
predictions = {name: prob[1] for name, prob in predictions}

for name, prob in predictions.items():
    st.write(f'{name} has a {prob} chance of death')
    
original_char_df = original_char_df[:original_shape[0]]