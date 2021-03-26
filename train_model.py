import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import numpy as np
import pickle

pickle_in = open("features.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("label.pickle","rb")
y = pickle.load(pickle_in)

X= np.array(X)
y= np.array(y)
X = X.reshape(X.shape[0],-1, X.shape[1])
#X = np.reshape(X, (X.shape[0], X.shape[1], 1))
model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)

model.add(LSTM(64, input_shape=(1,X.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(X,
          y,
          epochs=3,
          validation_split=0.1)