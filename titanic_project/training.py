import scipy
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, 'networks')

import datasets as data
import densenet as net

def next_batch(num, data, labels):
    idx = np.arange(0 , data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

n_epochs = 100
batch_size = 40
best_epoch = None
best_model = None

with tf.Session() as sess:
    net.init.run()
    acc_going_down = 0
    max_acc = 0
    for epoch in range(n_epochs):
        for iteration in range(data.titanic_prepared.shape[0] // batch_size):
            titanic_prepared_converted = scipy.sparse.csr_matrix.todense(data.titanic_prepared)
            survived_label_converted = data.survived_label.values
            X_batch, y_batch = next_batch(batch_size, titanic_prepared_converted, survived_label_converted)
            X_batch = np.squeeze(X_batch)
            if iteration % 10  == 0:
                summary_str = net.mse_summary.eval(feed_dict={net.X: X_batch, net.y: y_batch})
                step = epoch * (data.titanic_prepared.shape[0] // batch_size) + iteration
                net.file_writer.add_summary(summary_str, step)
            sess.run(net.training_op, feed_dict={net.X: X_batch, net.y: y_batch})
        acc_train = net.accuracy.eval(feed_dict={net.X: X_batch, net.y: y_batch})

        X_test = np.squeeze(scipy.sparse.csr_matrix.todense(data.titanic_test_prepared))
        Y_test = data.test_set["Survived"].values
        acc_val = net.accuracy.eval(feed_dict={net.X: X_test,
                                           net.y: Y_test})
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
        if (max_acc < acc_val):
            max_acc = acc_val
            acc_going_down = 0
            best_epoch = epoch
            save_path = net.saver.save(sess, "models/model4-23_02.ckpt")
        else:
            acc_going_down += 1
            if acc_going_down == 10:
                print("Early stop: Accuracy was going down")
                print("Epoch was: ", best_epoch)
                break

net.file_writer.close()
