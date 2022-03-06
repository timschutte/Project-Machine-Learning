import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
raw_data = pd.read_csv('tweets_labelled_09042020_16072020.csv', sep=";")
print('The shape of the raw data:', raw_data.shape)
#print(raw_data.head(10))
raw_text_short = np.array(raw_data['text'][:30])
#print(raw_text_short.shape)
#print(raw_text_short[:5])

def lower_case(data):
    new_data = np.array([])
    for tweet in data:
        new_data = np.append(new_data, tweet.lower())
    return new_data

def sentence_tokenize(data):            ###not sure if we need a sentence tokenizer, a sentence tokenizer is
    new_data = np.array([])             ###basically just splitting the different lines or paragraphs? (not
    for sent in data:                   ###sure which) into different list items. However, tweets dont contain
        sent = sent_tokenize(sent)      ###that many lines.
        new_data = np.append(new_data, sent)
    return new_data

def tokenize(data):
    new_data = np.array([])
    for line in data:
        line = word_tokenize(line)
        new_data = np.append(new_data, line)
    return new_data

def remove_characters(data):
    new_data = np.array([])
    for tweet in data:
        new_tweet = np.array([])
        for word in tweet:
            res = re.sub(r'[^\w\s]',"",word)
            if res != "":
                new_tweet = np.append(new_tweet, res)
        new_data = np.append(new_data, new_tweet)
    return new_data
data = lower_case(raw_text_short)
print(new_data[:3])