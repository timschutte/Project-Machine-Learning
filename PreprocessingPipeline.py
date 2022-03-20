import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk.stem as stemmers
import gensim
from sklearn import model_selection

class PreprocessPipeline():
    def __init__(self, X_raw=None, y_raw=None, language='english', stemmer='snowball', w2v_model=None, maxSequenceLength=70):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.language = language

        if stemmer == 'snowball':
            self.stemmer = stemmers.SnowballStemmer()
        elif stemmer == 'porter':
            self.stemmer = stemmers.PorterStemmer()
        elif stemmer == 'lancaster':
            self.stemmer = stemmers.LancasterStemmer()
        else:
            raise "The stemmer you entered is not recognized. The following keywords are supported:\n'snowball'\n'porter'\n'lancaster'"
        
        self.w2v_model = w2v_model
        self.X = None
        self.y = None
        self.y_mapping = None
        self.maxLen = maxSequenceLength

    def load_w2v(self, path):
        self.w2v_model = gensim.models.Word2Vec.load(path)
        
    def mapNanValues(self):
        nan_values = []
        for i in range(len(self.X_raw)):
            if type(self.X_raw[i]) is not str:
                nan_values.append(i)
        return nan_values

    def removeNanValues(self):
        nan_values = self.mapNanValues
        self.X_raw, self.y_raw = self.X_raw.drop(nan_values), self.y_raw.drop(nan_values)
        self.X_raw, self.y_raw = self.X_raw.reset_index(drop=True), self.y_raw.reset_index(drop=True)  

    def lowercase(self):
        new_data = []
        for tweet in self.X_raw:
            new_data.append(tweet.lower())
        self.X = new_data
    
    def tokenize(self):
        new_data = []
        for line in self.X:
            line = word_tokenize(line)
            new_data.append(line)
        self.X = new_data

    def remove_characters(self):
        new_data = []
        if type(self.X[0]) != list:
            raise "The first item of data is of type ", type(self.X[0]), ' the data is probably not tokenized yet.'
        for tweet in self.X:
            new_tweet = []
            for word in tweet:
                res = re.sub(r'[^\w\s]',"",word)
                if res != "":
                    new_tweet.append(res)
            new_data.append(new_tweet)
        self.X = new_data
    
    def remove_stopwords(self):
        new_data = []
        for tweet in self.X:
            new_tweet = []
            for word in tweet:
                if not word in stopwords.words(self.language):
                    new_tweet.append(word)
            new_data.append(new_tweet)
        self.X = new_data

    def stemming(self):
        new_data = []
        for tweet in self.X:
            new_tweet = []
            for word in tweet:
                new_tweet.append(self.stemmer.stem(word))
            new_data.append(new_tweet)
        self.X = new_data

    def lan_to_vec_sentence(self, sentence, maxLen, vector_size):
        vec_sentence = np.zeros((maxLen, vector_size))
        for i in range(min([len(sentence), maxLen])):
            try:
                vec_sentence[i] = self.w2v_model.wv[sentence[i]]
            except KeyError:
                continue
        return vec_sentence

    def lan_to_vec_dataset(self):
        vec_size = self.w2v_model.vector_size
        vec_dataset = np.zeros((len(self.X), self.maxLen, vec_size))
        for i in range(len(self.X)):
            vec_dataset[i] = self.lan_to_vec_sentence(self.X[i], vector_size=vec_size, maxLen=self.maxLen)
        self.X = vec_dataset

    def vectorizeY(self):
        unique_values = self.y_raw.unique()
        self.y_mapping = {index:unique_values[index] for index in range(len(unique_values))}
        self.y = np.zeros((len(self.y_raw), len(self.y_mapping)))
        for i in range(len(self.y_raw)):
            for index, value in self.y_mapping.items():
                if self.y_raw[i] == value:
                    self.y[i][index] = 1.0
                    break

    def train_test_split(self, test_size=0.3):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=0.33, random_state=35)
        self.X, self.y = {'train':X_train, 'test':X_test}, {'train':y_train, 'test':y_test}