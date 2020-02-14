import re
import os
import pickle
import pandas as pd
import preprocessor as p

import nltk
from nltk.tokenize import TweetTokenizer


with open(os.getcwd() + '/stopwords.json', 'rb') as f:
    stop_words = pickle.load(f)
tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)

def clean_text(t):
    p.set_options(
        p.OPT.URL,
        p.OPT.MENTION,
        p.OPT.HASHTAG,
        p.OPT.RESERVED,
        p.OPT.EMOJI,
        p.OPT.SMILEY,
    )
    t = p.clean(t)
    t = re.sub(r"[\\//_,;.:*+\-\>\<)(%^$|~&`'\"\[\]\=]+", '', t)
    t = re.sub(r'[^\x00-\x7F]+',' ', t)
    return t.lower()


def tokenize_text(t, tokenizer=None, stop_words=None):
    if not tokenizer:
        from nltk.tokenize import TweetTokenizer
        tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
        
    if not stop_words:
        with open(os.getcwd() + '/stopwords.json', 'rb') as f:
            stop_words = pickle.load(f)

    tList = [i for i in tokenizer.tokenize(t) if i not in stop_words]
    return tList


def load_embeding(path, max_length_dictionary=10000):
    embeddings_dict = {}
    i = 0
    
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            if values[0].isalnum():
                embeddings_dict[ values[0] ] = i
                i += 1

            if i == max_length_dictionary:
                break
                
    return embeddings_dict


def replace_token_with_index(tList, embeddingMap):
    tNewList = []
    for t in tList:
        # if t is not in EmbeddingMap continue the loop
        indice = embeddingMap.get(t)
        if not indice:
            continue
        else:
            tNewList.append(indice)
    return tNewList
    


