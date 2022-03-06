import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
port = PorterStemmer()
raw_data = pd.read_csv('tweets_labelled_09042020_16072020.csv', sep=";")
print('The shape of the raw data:', raw_data.shape)
#print(raw_data.head(10))
raw_text_short = raw_data['text']
#print(raw_text_short.shape)
#print(raw_text_short[:5])

def lower_case(data):
    new_data = []
    for tweet in data:
        new_data.append(tweet.lower())
    return new_data

def sentence_tokenize(data):            ###not sure if we need a sentence tokenizer, a sentence tokenizer is
    new_data = []      ###basically just splitting the different lines or paragraphs? (not
    for sent in data:                   ###sure which) into different list items. However, tweets dont contain
        sent = sent_tokenize(sent)      ###that many lines.
        new_data.append(sent)
    return new_data

def tokenize(data):
    new_data = []
    for line in data:
        line = word_tokenize(line)
        new_data.append(line)
    return new_data

def remove_characters(data):
    new_data = []
    for tweet in data:
        new_tweet = []
        for word in tweet:
            res = re.sub(r'[^\w\s]',"",word)
            if res != "":
                new_tweet.append(res)
        new_data.append(new_tweet)
    return new_data

def remove_stopwords(data):
    new_data = []
    for tweet in data:
        new_tweet = []
        for word in tweet:
            if not word in stopwords.words('english'):
                new_tweet.append(word)
        new_data.append(new_tweet)
    return new_data

def stemming(data):
    new_data = []
    for tweet in data:
        new_tweet = []
        for word in tweet:
            new_tweet.append(port.stem(word))
        new_data.append(new_tweet)
    return new_data



#####This is just to check if I don't fuck up the data#####
print('Step 0 shape:', len(raw_text_short))
print(raw_text_short[:3])
print('\n\n\n')
data = lower_case(raw_text_short)
print('Step 1 shape:', len(data))
print(data[:3])
print('\n\n\n')
data = tokenize(data)
print('Step 2 shape:', len(data))
print(data[:3])
print('\n\n\n')
data = remove_characters(data)
print('Step 3 shape:', len(data))
print(data[:3])
print('\n\n\n')
data = remove_stopwords(data)
print('Step 4 shape:', len(data))
print(data[:3])
print('\n\n\n')
data = stemming(data)
print('Step 5 shape:', len(data))
print(data[:3])
print('\n\n\n')

######  Word2Vec  ########
import gensim
model = gensim.models.Word2Vec(
    window = 3,
    min_count=2
)
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=5)
model.save("./fullDataset_model.model")