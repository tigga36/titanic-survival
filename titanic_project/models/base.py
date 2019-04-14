import tensorflow as tf
from functools import partial
import numpy as np

from prep import titanic_prepared
from prep import survived_label

n_inputs = 9
n_hidden1 = 100
n_hidden2 = 100
n_hidden3 = 100
n_hidden4 = 100
n_hidden5 = 100
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

he_init = tf.contrib.layers.variance_scaling_initializer()
my_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init)

with tf.name_scope("dnn1"):
    hidden1 = my_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_layer(hidden1, n_hidden2, name="hidden2")
    hidden3 = my_layer(hidden2, n_hidden3, name="hidden3")
    hidden4 = my_layer(hidden3, n_hidden4, name="hidden4")
    hidden5 = my_layer(hidden4, n_hidden5, name="hidden5")
    logits = my_layer(hidden5, n_outputs, name="output")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.001

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

n_epochs = 40
batch_size = 50
best_epoch = None
best_model = None
minimum_val_error = float("inf")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(titanic_prepared.shape[0] // batch_size):
            titanic_prepared_converted = scipy.sparse.csr_matrix.todense(titanic_prepared)
            survived_label_converted = survived_label.values
            X_batch, y_batch = next_batch(batch_size, titanic_prepared_converted, survived_label_converted)
            X_batch = np.squeeze(X_batch)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        
        X_test = np.squeeze(scipy.sparse.csr_matrix.todense(titanic_test_prepared))
        Y_test = test_set["Survived"].values
        acc_val = accuracy.eval(feed_dict={X: X_test,
                                          y: Y_test})
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
        if (1-acc_val < minimum_val_error):
            minimum_val_error = 1-acc_val
            best_epoch = epoch
    
    save_path = saver.save(sess, "./my_model_final.ckpt")