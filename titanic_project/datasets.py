import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

os.path.abspath(os.path.join(yourpath, os.pardir))
TITANIC_PATH = os.path.join("../data/raw", "titanic")

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

#============================================================
#Start cleaning test data

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Sex", "Embarked"]

train_set_pre = train_set.copy()
train_set_pre["Sex"] = pd.get_dummies(train_set["Sex"])
train_set_pre["Embarked"] = pd.get_dummies(train_set["Embarked"])
train_set_pre["Embarked"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(missing_values=np.nan,strategy="median")),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

titanic_prepared = full_pipeline.fit_transform(train_set_pre)
