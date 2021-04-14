import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import numpy as np
import pickle
from sklearn import preprocessing
from numpy.random import seed
seed(2)
pickle_in = open("features.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("label.pickle","rb")
y = pickle.load(pickle_in)

X= np.array(X)
y= np.array(y)
X = np.fft.fft(X)
X = np.absolute(X)
#scaler = preprocessing.StandardScaler().fit(X) #standard scaling
#X = scaler.transform(X)
X = X.reshape(X.shape[0],-1, X.shape[1])
print(X.shape)
#X = np.reshape(X, (X.shape[0], X.shape[1], 1))
model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)

model.add(LSTM(128, input_shape=(X.shape[1],X.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
INIT_LR = 1e-4
EPOCHS = 5
#opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt=tf.keras.optimizers.Adagrad(
    learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
    name='Adagrad'
)

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(X,
          y,
          epochs=5,
          validation_split=0.1)