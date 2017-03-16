import tensorflow as tf
import numpy as np
import os
import shutil


X_train = np.load("training_set.npy")
y_train = np.load("training_labels.npy")


# Construction phase
tf.reset_default_graph()

# Let m be the number of training examples and n the number of features.
m, n = X_train.shape

n_hidden1 = 50               # Neurons for first hidden layer
n_hidden2 = 50               # Neurons for second hidden layer
n_outputs = 18               # Output labels
alpha = 0.01                 # The learning rate

X = tf.placeholder(tf.float32, shape=[None, n], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")

from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits  = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

# NOTE
# What's the difference between sparse_softmax_cross_entropy_with_logits() and
# softmax_cross_entropy_with_logits()? Why use cross entropy at all?
#
# Cross entropy will penalize models the estimate low probability for
# the target class. The implementation we will use is Tensorflow's
# sparse_softmax_cross_entropy_with_logits(). It computes the cross entropy
# based on the output of the network BEFORE going through the softmax activation
# function. It expects labels in the form of integers ranging from 0 to
# the number of classes minus 1. In our case that means 0-17.
#
# Answer found on StackOverflow:
# ==============================
#
#   The difference is simple:
#
#     For sparse_softmax_cross_entropy_with_logits, labels must have the shape
#     [batch_size] and the dtype int32 or int64. Each label is an int in range
#     [0, num_classes).
#
#     For softmax_cross_entropy_with_logits, labels must have the shape
#     [batch_size, num_classes] and dtype float32 or float64.
#
#   Labels used in softmax_cross_entropy_with_logits are the one hot version
#   of labels used in sparse_softmax_cross_entropy_with_logits.
#
#   Another tiny difference is that with sparse_softmax_cross_entropy_with_logits,
#   you can give -1 as a label to have loss 0 on this label.
#
# See: http://stackoverflow.com/questions/37312421
#
with tf.name_scope("loss"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Execution Phase

init = tf.global_variables_initializer()

# This neural network begins to completely overfit the training set
# around 2300 iterations
n_epochs   = 3000
batch_size = 50
n_batches  = int(np.ceil(m / batch_size))

# fetch_batch() returns a subset of our training set which we will
# use to perform one step of gradient descent.
#
# :epoch:
# An epoch is an iteration through the training set.
#
# :batch_index:
# The batch_index determines where in the training set we pluck examples from.
#
# :batch_size:
# batch_size is the number iterations we run through the training set per epoch
def fetch_batch(epoch, batch_index, batch_size):
    """
    In mini-batch gradient descent we take a small number of random
    examples to perform one step of gradient descent. This is in contrast
    to batch gradient descent which uses the entire training set to
    perform one step.
    """
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size, dtype=np.int32)
    X_batch = X_train[indices]
    y_batch = y_train[indices]
    return (X_batch, y_batch)

# Create tmp and results directories so we can
# save off checkpoints and the final model to disk.
saver       = tf.train.Saver()
tmp_dir     = "tmp"
results_dir = "results"

if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)

if os.path.isdir(results_dir):
    shutil.rmtree(results_dir)

os.mkdir(tmp_dir)
os.mkdir(results_dir)

# Every 50 epochs we will test our model with clean data
X_test = np.load("test_set.npy")
y_test = np.load("test_labels.npy")


with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        # Save every 100 eopchs
        if epoch % 500 == 0:
            save_path = saver.save(sess, tmp_dir + "/poke_model.ckpt")

        for batch_index in range(n_batches):
            # Find next batch then run gradient descent
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        # Shows progress every 10 epochs
        if epoch % 50 == 0:
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    # Save the final model
    save_path = saver.save(sess, results_dir + "/poke_model_final.ckpt")
