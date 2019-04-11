import os
import pandas as pd
import numpy as np

TITANIC_PATH = os.path.join("raw", "titanic")

def load_titanic_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path,"train.csv")
    return pd.read_csv(csv_path)

titanic = load_titanic_data()

#random seed
np.random.seed(21)

#Splitting the training data (80% for training, 20% for testing)
def split_train_test(data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]

train_set, test_set = split_train_test(titanic, 0.2)

#Sex was a string value, numerize to better fit with network
titanic_cat = titanic["Sex"]
titanic_cat_encoded, titanic_cat = titanic_cat.factorize()
titanic_mod = titanic
titanic_mod["Sex"] = titanic_cat_encoded
titanic_mod["Sex"].value_counts()