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
raw_text_short = raw_data['text']

##### Preprocessing functions #####
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

###################################################################
################# Actual Preprocessing ############################
#####The print statements to check if I don't fuck up the data#####

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



### Hyperparameters ###
def longest_tweet(tweets):      #To check for the longest sentence in the dataset
    maxLen = 0
    for tweet in tweets:
        if len(tweet) > maxLen:
            maxLen = len(tweet)
    return maxLen
    
windowSize = 5
vectorSize = 100
maxLen = 50

######  Word2Vec  ########

# creating a vector representation of the words in the dataset
import gensim
"""
model = gensim.models.Word2Vec(
    window = windowSize,
    min_count=2,
    vector_size = vectorSize
)
model.build_vocab(data)
model.train(data, total_examples=model.corpus_count, epochs=5)
model.save("./fullDataset_model.model")
"""
model = gensim.models.Word2Vec.load("fullDataset_model.model")
word_vectors = model.wv

###### Converting a natural language sentence into a vector representation suitable for the keras network

def lan_to_vec_sentence(sentence, vector_size=vectorSize, maxLen=maxLen):
    vec_sentence = np.zeros((maxLen, vector_size))
    for i in range(min([len(sentence), maxLen])):
        try:
            vec_sentence[i] = word_vectors[sentence[i]]
        except KeyError:
            continue
    return vec_sentence

def lan_to_vec_dataset(dataset, vector_size=vectorSize, maxLen=maxLen):
    vec_dataset = np.zeros((len(dataset), maxLen, vector_size))
    for i in range(len(dataset)):
        vec_dataset[i] = lan_to_vec_sentence(dataset[i], vector_size=vector_size, maxLen=maxLen)
    return vec_dataset

vectorized = lan_to_vec_dataset(data[:10])
print(vectorized.shape, '\n\n\n')
print(vectorized[:2])

"""
keyedvectors = gensim.models.KeyedVectors.load("fullDataset_model.model", mmap='r')
embedding_matrix = np.zeros((vocab_len + 1, vectorSize))
print(type(keyedvectors[0]))
print(keyedvectors[0])
for word, i in word_vectors.word_to_key:
    embedding_vector = word_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
"""
#### LSTM Model ####
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Softmax
from keras.layers.embeddings import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
embedding_layer = Embedding(input_dim=vocab_len, output_dim=vectorSize, input_length=maxLen, weights = [embedding_matrix], trainable=False)

lstm_model = Sequential()
"""
