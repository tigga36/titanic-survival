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
test_real["Toddler"] = np.where(test_real["Age"]<=5, 1, 0)
test_real["YoungMale"] = np.where((test_real["Sex"]=='male')&(test_real["Age"]>=15)&(test_real["Age"]<=35), 1, 0)
test_real["AgeRange"] = 1
test_real["AgeRange"][(test_real["Age"] > 60)] = 3
test_real["AgeRange"][(test_real["Age"] < 15)] = 0
test_real["FareRange"] = 0
test_real["FareRange"][(test_real["Fare"] > 50)] = 1
test_real["FareRange"][(test_real["Fare"] > 100)] = 2
test_real["FareRange"][(test_real["Fare"] > 200)] = 3

test_real["Fam"] = test_real["SibSp"] + test_real["Parch"]

TitleDict = {
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss"
}

def get_titles():
    test_real['Title'] = test_real['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    test_real['Title'] = test_real.Title.map(TitleDict)
    return test_real


test_real = get_titles()


def process_names():
    global test_real

    titles_dummies = pd.get_dummies(test_real['Title'], prefix='Title')
    test_real = pd.concat([test_real, titles_dummies], axis=1)

    test_real.drop('Title', axis=1, inplace=True)

    return test_real


test_real = process_names()

test_real_pre = test_real.copy()
test_real_pre["Sex"] = pd.get_dummies(test_real["Sex"])
test_real_pre["Embarked"] = pd.get_dummies(test_real["Embarked"])

test_real_pre = test_real_pre.drop(['PassengerId', 'Fare', 'Name', 'Ticket', 'Cabin', 'Age', 'Parch', 'SibSp', 'Embarked', 'Title_Mr'], axis=1)

test_real_prepared = data.full_pipeline.fit_transform(test_real_pre)

with tf.Session() as sess:
    net.saver.restore(sess, "./models/model_acc0.8202247.ckpt")
    # test_real_prepared_converted = scipy.sparse.csr_matrix.todense(test_real_prepared)
    test_real_prepared_converted = (test_real_prepared)
    X_new_scaled = np.asarray(test_real_prepared_converted)
    X_new_scaled = np.squeeze(X_new_scaled)
    Z = net.logits.eval(feed_dict={net.X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

result = pd.DataFrame(data= y_pred, index=test_real["PassengerId"], columns=None, dtype=None, copy=False)
result.columns = ['Survived']
result.to_csv('../results/results4-26_05.csv')