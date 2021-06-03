import pandas as pd
import numpy as np
import re

from nltk import RegexpTokenizer
from contractions import contractions_dict

import shakespeare_dicts as sd

def get_stopwords():
    return ['ye', 'ay']

def clean_contractions(sentence):
    
    contractions = contractions_dict.copy()
    contractions = {re.sub(r'’', "'", key): re.sub(r'’', "'", value) for key, value in contractions.items() if '.' not in key}
    contractions = {key.lower():value for key, value in contractions.items() if "'" in key}
    
    shakespeare_contractions_dict = sd.shakespeare_contractions_dict()
    
    all_dicts = [contractions, shakespeare_contractions_dict]
    
    sentence = clean_punctuation(sentence)
    new_sentence = ""
    
    tokenizer = RegexpTokenizer('\S+')

    for word in tokenizer.tokenize(sentence):
        word = word.lower()

        for dicto in all_dicts:
            if word in dicto:
                word = dicto[word]
 
        new_sentence += word + ' '

    return new_sentence

def clean_anachronisms(sentence):
    anachronisms = sd.anachronisms_dict()
    orings = sd.oring_dict()
    ofer = sd.ofer_dict()
    ths = sd.th_dict()
    sts = sd.st_dict()
    
    all_dicts = [anachronisms, orings, ths, sts, ofer]
    
    tokenizer = RegexpTokenizer('\S+')

    sentence = clean_punctuation(sentence)
    new_sentence = ""

    for word in tokenizer.tokenize(sentence):
        word = word.lower()

        for dicto in all_dicts:
            if word in dicto:
                word = dicto[word]
 
        new_sentence += word + ' '

    return new_sentence

def clean_punctuation(sentence):
    sentence = re.sub(r"(?<=\A)([.?!-,:;\"\-])+|([.?!-,:;\"\-])+(?=\Z)", ' ', sentence)
    sentence = re.sub(r"(?<=\s)([.?!-,:;\"\-])+|([.?!-,:;\"\-])+(?=\s)", ' ', sentence)
    return re.sub(r"([.?!-,:;\"\-]){2,}", ' ', sentence) 

def corpusize(df, column):
    tokenizer = RegexpTokenizer('\S+')
    corpus = {}

    for line in df[column].values:
        #strip excess puncutation
        line = clean_punctuation(line)
        for word in tokenizer.tokenize(line):
            word = word.lower()
            if word not in corpus:
                corpus[word] = 1
            else:
                corpus[word] += 1

    return {k: v for k, v in sorted(corpus.items(), key=lambda item: item[1], reverse = True)}