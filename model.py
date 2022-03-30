import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Dropout, Input, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import tensorflow_text as text
from PreprocessingPipeline import PreprocessPipeline

def mapNanValues(data):
    nan_values = []
    for i in range(len(data)):
        if type(data) is not str:
            nan_values.append(i)
    return nan_values

X = pd.read_csv('Twitter_Data.csv')
X = X['clean_text']
X = X.drop(mapNanValues(X)).reset_index()
y = pd.read_csv('preprocessedY.csv', index_col=0)
X = X[:100]
y = y[:100]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

if X.shape[0] != y.shape[0]:
    print("Theres something wrong with the datashape")

METRICS = [
      CategoricalAccuracy(name='accuracy'),
      Precision(name='precision'),
      Recall(name='recall')
]

# X = pd.read_csv('preprocessedX.csv', index_col=0)
# dim = X.shape
# X = np.array(X).reshape((int(dim[0]/40), 40, 100))


def Vanilla_LSTM():
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(40, 100), return_sequences=True))
    lstm_model.add(LSTM(128, return_sequences=False))
    lstm_model.add(Dense(32))
    lstm_model.add(Dense(3))
    lstm_model.add(Softmax(input_shape=[0., 0., 0.]))
    print('\n\nmodel built')
    lstm_model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(0.01))
    print('\n\nmodel compiled')
    lstm_model.fit(x=X_train, y=y_train, batch_size=5, epochs=10, verbose=1, validation_split=0.1)

def Homemade_LSTM():
    preprocessor = PreprocessPipeline()
    text_input = Input(shape=(), dtype=tf.string, name='text')
    preprocessed = preprocessor(X_raw=text_input).unprocessed_to_vector()
    l = LSTM(128, shape=(40, 100), activation='relu', return_sequences=True)(preprocessed)
    l = Dropout(0.1)(l)
    l = LSTM(128, activation='relu', return_sequences=False)(l)
    l = Dropout(0.1)(l)
    l = Dense(32)(l)
    output = Softmax(3)(l)
    model = Model(inputs=[text_input], outputs = [output])
    return model
 
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def Bert_LSTM():
    text_input = Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    l = LSTM(128, shape=(40, 100), activation='relu', return_sequences=True)(outputs['pooled_output'])
    l = Dropout(0.1)(l)
    l = LSTM(128, activation='relu', return_sequences=False)(l)
    l = Dropout(0.1)(l)
    l = Dense(32)(l)
    output = Softmax(3)(l)
    model = Model(inputs=[text_input], outputs = [output])
    return model

def Trainable_Embedding_LSTM():
    text_input = Input(shape=(), dtype=tf.string, name='text')
    intencoding = one_hot(text_input, 999)
    padded = pad_sequences(intencoding, maxlen=40, padding='post')
    l = Embedding(input_dim=999, output_dim=100, input_length=40)
    l = LSTM(128, shape=(40, 100), activation='relu', return_sequences=True)(l)
    l = Dropout(0.1)(l)
    l = LSTM(128, activation='relu', return_sequences=False)(l)
    l = Dropout(0.1)(l)
    l = Dense(32)(l)
    output = Softmax(3)(l)
    model = Model(inputs=[text_input], outputs = [output])
    return model


model1 = Homemade_LSTM()
model2 = Bert_LSTM()
model3 = Trainable_Embedding_LSTM()

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
model1.fit(X, y, batch_size = 10, verbose=1)

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
model2.fit(X, y, batch_size = 10, verbose=1)

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
model3.fit(X, y, batch_size = 10, verbose=1)
