import re
import pandas as pd
import preprocessor as p

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


stop_words = set(stopwords.words('english'))
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
    return t


def tokenize_text(t, tokenizer=None, stop_words=None):
    if not tokenizer:
        from nltk.tokenize import TweetTokenizer
        tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
        
    if not stop_words:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

    tList = [i for i in tokenizer.tokenize(t) if i not in stop_words]
    return tList


def replace_token_with_index(tList):
    


import os
df = pd.read_csv(os.getcwd() + '/eng_twitter.csv')
print(df['text'].iloc[:3].apply(lambda x: tokenize_text(clean_text(x), tokenizer=tknzr, stop_words=stop_words)).values.tolist())