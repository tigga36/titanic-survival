import tensorflow as tf
from functools import partial

from datetime import datetime
import pytz

now = datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_neurons = 40

n_inputs = 7
n_hidden1 = n_neurons
n_hidden2 = n_neurons
n_hidden3 = n_neurons
n_hidden4 = n_neurons
n_hidden5 = n_neurons
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.8
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

scale = 0.001
he_init = tf.contrib.layers.variance_scaling_initializer()
my_layer_pre = partial(tf.layers.dense, kernel_initializer=he_init, kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
my_batch_norm = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
my_batch_drop = partial(tf.layers.dropout, dropout_rate, training=training)

with tf.name_scope("dnn1"):
    hidden1_pre = my_layer_pre(X_drop, n_hidden1, name="hidden1")
    hidden1_batch = my_batch_norm(hidden1_pre)
    hidden1_drop = tf.layers.dropout(hidden1_batch, dropout_rate, training=training)
    hidden1_act = tf.nn.elu(hidden1_drop)

    hidden2_pre = my_layer_pre(hidden1_act, n_hidden2, name="hidden2")
    hidden2_batch = my_batch_norm(hidden2_pre)
    hidden2_drop = tf.layers.dropout(hidden2_batch, dropout_rate, training=training)
    hidden2_act = tf.nn.elu(hidden2_drop)

    hidden3_pre = my_layer_pre(hidden2_act, n_hidden3, name="hidden3")
    hidden3_batch = my_batch_norm(hidden3_pre)
    hidden3_drop = tf.layers.dropout(hidden3_batch, dropout_rate, training=training)
    hidden3_act = tf.nn.elu(hidden3_drop)

    hidden4_pre = my_layer_pre(hidden3_act, n_hidden4, name="hidden4")
    hidden4_batch = my_batch_norm(hidden4_pre)
    hidden4_drop = tf.layers.dropout(hidden4_batch, dropout_rate, training=training)
    hidden4_act = tf.nn.elu(hidden4_drop)

    hidden5_pre = my_layer_pre(hidden4_act, n_hidden5, name="hidden5")
    hidden5_batch = my_batch_norm(hidden5_pre)
    hidden5_drop = tf.layers.dropout(hidden5_batch, dropout_rate, training=training)
    hidden5_act = tf.nn.elu(hidden5_drop)

    logits_pre = tf.layers.dense(hidden5_act, n_outputs, name="output")
    logits = my_batch_norm(logits_pre)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.001
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

mse_summary = tf.summary.scalar('accuracy', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())