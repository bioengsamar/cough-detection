#from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tff

tff.reset_default_graph()
n_steps = 1
n_inputs = 320
n_neurons = 150
n_outputs = 2
learning_rate = 0.001
X = tff.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tff.placeholder(tf.int32, [None])
basic_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tff.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
logits = tf.compat.v1.layers.dense(states, n_outputs, activation=None)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
labels=y, logits=logits)
#print(K.learning_phase().dtype)
loss = tf.reduce_mean(xentropy)
optimizer = tff.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tff.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tff.global_variables_initializer()

pickle_in = open("features.pickle","rb")
data = pickle.load(pickle_in)

pickle_in = open("label.pickle","rb")
labels = pickle.load(pickle_in)

data= np.array(data).astype(np.float32)
labels= np.array(labels).astype(np.int32)
data = data.reshape(data.shape[0],-1, data.shape[1])
(x_train, x_test, y_train, y_test) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
n_epochs = 5
batch_size = 32

with tff.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(x_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: x_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)