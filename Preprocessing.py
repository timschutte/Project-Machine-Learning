import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
raw_data = pd.read_csv('tweets_labelled_09042020_16072020.csv', sep=";")
print('The shape of the raw data:', raw_data.shape)
print(raw_data.head(10))
raw_text_short = np.array(raw_data['text'][:30])
print(raw_text_short.shape)
print(raw_text_short[:5])

def lower_case(data):
    new_data = np.array()
    for tweet in data:
        new_data.append(tweet.lower())
    return new_data

