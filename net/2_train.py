import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime


X_train = np.load("training_set.npy")
y_train = np.load("training_labels.npy")

# Construction phase
tf.reset_default_graph()

# Let m be the number of training examples and n the number of features.
m_examples, n_features = X_train.shape

n_neurons = 20               # Number of neurons in hidden layers
n_outputs = 2                # Output labels
alpha = 0.01                 # The learning rate

X = tf.placeholder(tf.float32, shape=[None, n_features], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")

from tensorflow.contrib.layers import fully_connected
# SEE: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_neurons, scope="hidden1", activation_fn=tf.nn.relu)
    logits  = fully_connected(hidden1, n_outputs, scope="outputs", activation_fn=None)


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
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct  = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Execution Phase

init = tf.global_variables_initializer()

# This neural network begins to completely overfit the training set
# around 2300 iterations
n_epochs   = 10000
batch_size = 640
n_batches  = int(np.ceil(m_examples / batch_size))

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
    indices = np.random.randint(m_examples, size=batch_size, dtype=np.int32)
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
X_val = np.load("validation_set.npy")
y_val = np.load("validation_labels.npy")

# Setup log directory for Tensorboard to read from
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_log_dir = "tf_logs"
log_dir = "{}/run-{}".format(root_log_dir, now)

# Implement Tensorboard
loss_summary   = tf.summary.scalar("ENTROPY", loss)
summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

# NOTE
# In order to monitor GPU usage in real time, use the
# watch command with the arguments below:
#
#       watch -n 5 nvidia-smi -a --display=utilization
#
with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):

        for batch_index in range(n_batches):
            # Find next batch then run gradient descent
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            # Log training stats for Tensorboard
            if batch_index % 10 == 0:
                summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)

        if epoch % 300 == 0:
            # Print progress report out to console
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val   = accuracy.eval(feed_dict={X: X_val, y: y_val})
            print("Epoch:", epoch,
                  "| Train acc:", acc_train,
                  "| Validation acc:", acc_val)

            # TODO
            # Right now we're saving a checkpoint every 300 epochs, but
            # it would be better to save only if the model is doing better.
            save_path = saver.save(sess, tmp_dir + "/poke_model.ckpt")

            print("====================\n")


    # Save the final model
    save_path = saver.save(sess, results_dir + "/poke_model_final.ckpt")


print("Finished training.")
