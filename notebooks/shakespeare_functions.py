import pandas as pd
import numpy as np
import re

from nltk import RegexpTokenizer
import shakespeare_dicts as sd

def clean_anachronisms(sentence):
    anachronisms = {'o': 'oh', 'hath': 'has', 'thou': 'you', 
                    'thy': 'your', 'doth': 'does', 'tis': 'it is', 'ere': 'before', 'shalt': 'will',
                    'thyself': 'yourself', 'ofer': 'over', 'twere': 'it were', 'hateth': 'hates',
                    'handicraftman': 'craftsman', 'cavalery': 'cavalry',
                    'alack': 'alas', 'eyne': 'eyes', 'wot': 'what',
                    'persevere': 'persever',  'witnesseth': 'witness',
                    'ofercharged': 'overcharged', 'momentany': 'momentary',
                    'watory': 'watery', 'oferlook': 'overlook', 'nointed': 'anointed', 'oferrules': 'overrules',
                    'flewed': 'flew', 'leathern': 'leathery',
                    'yond': 'yonder', 'whatsomever': 'whatsoever', 'twas': 'it was',
                    'twould': 'it would', 'threatoningly': 'threateningly', 
                    'ofertaken': 'overtaken', 'ofercount': 'overcount', 
                    'ofertake': 'overtake',  'tween': 'between', 
                    'twixt': 'between', 'importeth': 'import',  'holp': 'hope', 'encountoring': 'encountering',
                    'sdeath': 'god death', 'sblood': 'god blood', 'oferpeer': 'overlook',
                    'unproperly': 'improperly', 
                    'underpeep': 'underlook', 'oferlaboured': 'overworked',
                    'madded': 'enraged', 
                    'sufficeth': 'suffices',  'oferdoing': 'overdoing', 
                    'oferdone': 'overdone', 'oferstep': 'overstep',
                    'noyance': 'annoyance', 
                    'wisheth': 'wishes', 'doteth': 'dotes',  'oferpaid': 'overpayed',
                    'oferpeered': 'overlooked'
                    }
    tokenizer = RegexpTokenizer('\S+')

    sentence = clean_punctuation(sentence)
    new_sentence = ""

    for word in tokenizer.tokenize(sentence):
        word = word.lower()

        if word in anachronisms:
            word = anachronisms[word]

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