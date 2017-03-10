import tensorflow as tf
import numpy as np

# Load a trained model and evaluate it.

X_test = np.load("test_set.npy")
y_test = np.load("test_labels.npy")

# Construction Phase

tf.reset_default_graph()

# Let m be the number of training examples and n the number of features.
m, n = X_test.shape

n_hidden1 = 300               # Neurons for first hidden layer
n_hidden2 = 300               # Neurons for second hidden layer
n_outputs = 18               # Output labels
alpha = 0.01                 # The learning rate

X = tf.placeholder(tf.float32, shape=[None, n], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")

from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits  = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Execution Phase
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./results/poke_model_final.ckpt")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Test accuracy:", acc_test)
