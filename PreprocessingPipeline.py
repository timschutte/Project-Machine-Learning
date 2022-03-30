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

    def __init__(self, X_raw=None, y_raw=None, language='english', stemmer='snowball', w2v_model=None, maxSequenceLength=40):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.language = language

        if stemmer == 'snowball':
            self.stemmer = stemmers.SnowballStemmer(language)
        elif stemmer == 'porter':
            self.stemmer = stemmers.PorterStemmer(language)
        elif stemmer == 'lancaster':
            self.stemmer = stemmers.LancasterStemmer(language)
        else:
            raise "The stemmer you entered is not recognized. The following keywords are supported:\n'snowball'\n'porter'\n'lancaster'"
        
        self.w2v_model = w2v_model
        self.X = None
        self.y = None
        self.y_mapping = None
        self.maxLen = maxSequenceLength
        self.typeErrors = []

    def load_w2v(self, path):
        self.w2v_model = gensim.models.Word2Vec.load(path)
        
    def mapNanValues(self):
        nan_values = []
        for i in range(len(self.X_raw)):
            if type(self.X_raw[i]) is not str:
                nan_values.append(i)
        return nan_values

    def removeNanValues(self):
        nan_values = self.mapNanValues()
        if len(nan_values) > 0:
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
            try:
                new_line = word_tokenize(line)
                new_data.append(new_line)
            except TypeError:
                self.typeErrors.append(str(type(line)) + '    ' +  str(line))
        print(self.typeErrors)
        self.X = new_data

    def remove_characters(self):
        new_data = []
        if type(self.X[0]) != list:
            raise "The first item of data is of type " + str(type(self.X[0])) + ' the data is probably not tokenized yet.'
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
    
    def to2demensions(self):
        dimension = self.X.shape
        return self.X.reshape((dimension[0]*dimension[1], dimension[2]))

    def save_data(self):
        pd.DataFrame(self.y).to_csv('preprocessedY.csv')
        pd.DataFrame(self.to2demensions()).to_csv('preprocessedX.csv')
        mapping = 'The following indexes in the y vectors encode for the following classes:'
        for index, value in self.y_mapping.items():
            mapping += '\n'+ str(index) + ' : ' + str(value)
        text_file = open("mapping.txt", "w")
        text_file.write(mapping)
        text_file.close()


    def train_test_split(self, test_size=0.3):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=test_size, random_state=35)
        self.X, self.y = {'train':X_train, 'test':X_test}, {'train':y_train, 'test':y_test}

    def unprocessed_to_unembedded(self):
        self.removeNanValues()
        self.tokenize()
        self.remove_characters()
        self.remove_stopwords()
        self.stemming()
        self.vectorizeY()

    def unprocessed_to_vector(self):
        self.removeNanValues()
        self.lowercase()
        self.tokenize()
        self.remove_characters()
        self.remove_stopwords()
        self.stemming()
        self.lan_to_vec_dataset()
        return self.X

data = pd.read_csv('Twitter_Data.csv')
x, y = data['clean_text'], data['category']
pipeline = PreprocessPipeline(X_raw=x, y_raw=y)
# pipeline.load_w2v('tweets_w2v_2.model')
# pipeline.unprocessed_to_vector()
# pipeline.save_data()
