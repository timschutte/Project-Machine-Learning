import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Softmax
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
X = pd.read_csv('preprocessedX.csv', index_col=0)
dim = X.shape
X = np.array(X).reshape((int(dim[0]/70), 70, 100))
y = pd.read_csv('preprocessedY.csv', index_col=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(70, 100), return_sequences=True))
lstm_model.add(LSTM(128, return_sequences=False))
lstm_model.add(Dense(32))
lstm_model.add(Dense(3))
lstm_model.add(Softmax(input_shape=[0., 0., 0.]))
print('\n\n\nmodel built')
lstm_model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(0.01))
print('\n\n\nmodel compiled')
lstm_model.fit(x=X_train, y=y_train, batch_size=5, epochs=10, verbose=1, validation_split=0.1)