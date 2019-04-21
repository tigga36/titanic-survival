import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import scipy

sys.path.insert(0, 'networks')
import datasets as data
import densenet as net


def load_titanic_test(titanic_path=data.TITANIC_PATH):
    csv_path = os.path.join(titanic_path,"test.csv")
    return pd.read_csv(csv_path)
test_real = load_titanic_test()

test_real_pre = test_real.copy()
test_real_pre["Sex"] = pd.get_dummies(test_real["Sex"])
test_real_pre["Embarked"] = pd.get_dummies(test_real["Embarked"])

test_real_prepared = data.full_pipeline.fit_transform(test_real_pre)

test_real_prepared

with tf.Session() as sess:
    net.saver.restore(sess, "./models/model4-21_01.ckpt")
    test_real_prepared_converted = scipy.sparse.csr_matrix.todense(test_real_prepared)
    X_new_scaled = np.asarray(test_real_prepared_converted)
    X_new_scaled = np.squeeze(X_new_scaled)
    Z = net.logits.eval(feed_dict={net.X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

result = pd.DataFrame(data= y_pred, index=test_real["PassengerId"], columns=None, dtype=None, copy=False)
result.columns = ['Survived']
result.to_csv('../results/results4-21_01.csv')