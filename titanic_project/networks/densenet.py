import tensorflow as tf
from functools import partial
import numpy as np
import scipy

n_inputs = 9
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 50
n_hidden4 = 25
n_hidden5 = 10
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.65
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

he_init = tf.contrib.layers.variance_scaling_initializer()
my_layer_pre = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init)
my_batch_norm = partial(tf.layers.batch_normalization, training=training, momentum=0.99)
my_batch_drop = partial(tf.layers.dropout, dropout_rate, training=training)

with tf.name_scope("dnn1"):
    hidden1_pre = my_layer_pre(X_drop, n_hidden1, name="hidden1")
    hidden1_batch = my_batch_norm(hidden1_pre)
    hidden1_drop = tf.layers.dropout(hidden1_batch, dropout_rate, training=training)
    hidden2_pre = my_layer_pre(hidden1_drop, n_hidden2, name="hidden2")
    hidden2_batch = my_batch_norm(hidden2_pre)
    hidden2_drop = tf.layers.dropout(hidden2_batch, dropout_rate, training=training)
    hidden3_pre = my_layer_pre(hidden2_drop, n_hidden3, name="hidden3")
    hidden3_batch = my_batch_norm(hidden3_pre)
    hidden3_drop = tf.layers.dropout(hidden3_batch, dropout_rate, training=training)
    hidden4_pre = my_layer_pre(hidden3_drop, n_hidden4, name="hidden4")
    hidden4_batch = my_batch_norm(hidden4_pre)
    hidden4_drop = tf.layers.dropout(hidden4_batch, dropout_rate, training=training)
    hidden5_pre = my_layer_pre(hidden4_drop, n_hidden5, name="hidden5")
    hidden5_batch = my_batch_norm(hidden5_pre)
    hidden5_drop = tf.layers.dropout(hidden5_batch, dropout_rate, training=training)
    logits_pre = my_layer_pre(hidden5_drop, n_outputs, name="output")
    logits = my_batch_norm(logits_pre)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.0005
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()