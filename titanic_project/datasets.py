import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

TITANIC_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/titanic/")

def load_titanic_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path,"train.csv")
    return pd.read_csv(csv_path)

titanic = load_titanic_data()

#checking for missing values

#MODIFICATIONS, ATTRIBUTE COMBINATIONS

# titanic["Toddler"] = np.where(titanic["Age"]<=5, 1, 0)
titanic["YoungMale"] = np.where((titanic["Sex"]=='male')&(titanic["Age"]>=15)&(titanic["Age"]<=35), 1, 0)
titanic["AgeRange"] = 1
titanic["AgeRange"][(titanic["Age"] > 60)] = 2
titanic["AgeRange"][(titanic["Age"] < 15)] = 0
titanic["Fam"] = titanic["SibSp"] + titanic["Parch"]
titanic["FareRange"] = 0
titanic["FareRange"][(titanic["Fare"] > 50)] = 1
titanic["FareRange"][(titanic["Fare"] > 100)] = 2
titanic["FareRange"][(titanic["Fare"] > 150)] = 3
titanic["FareRange"][(titanic["Fare"] > 200)] = 4

TitleDict = {
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss"
}

def get_titles():
    titanic['Title'] = titanic['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    titanic['Title'] = titanic.Title.map(TitleDict)
    return titanic


titanic = get_titles()


def process_names():
    global titanic

    titles_dummies = pd.get_dummies(titanic['Title'], prefix='Title')
    titanic = pd.concat([titanic, titles_dummies], axis=1)

    titanic.drop('Title', axis=1, inplace=True)

    return titanic


titanic = process_names()

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


num_attribs = ["Pclass", "AgeRange", "FareRange", "Fam"]
cat_attribs = ["Sex", "Title_Miss", "Title_Mrs"]

train_set_pre = train_set.copy()
train_set_pre["Sex"] = pd.get_dummies(train_set["Sex"])
train_set_pre["Embarked"] = pd.get_dummies(train_set["Embarked"])

test_set_pre = test_set.copy()
test_set_pre["Sex"] = pd.get_dummies(test_set["Sex"])
test_set_pre["Embarked"] = pd.get_dummies(test_set["Embarked"])

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(missing_values=np.nan,strategy="median")),
    ('std_scaler', StandardScaler())
    # ('minmax_scaler', MinMaxScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    # ('cat_encoder', OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

train_survived_label = train_set_pre["Survived"]
test_survived_label = test_set_pre["Survived"]
train_set_pre = train_set_pre.drop(['PassengerId', 'Fare', 'Survived', 'Name', 'Ticket', 'Cabin', 'Age', 'Parch', 'SibSp', 'Embarked', 'Title_Mr'], axis=1)
test_set_pre = test_set_pre.drop(['PassengerId', 'Fare', 'Survived', 'Name', 'Ticket', 'Cabin', 'Age', 'Parch', 'SibSp', 'Embarked', 'Title_Mr'], axis=1)

titanic_prepared = full_pipeline.fit_transform(train_set_pre)
titanic_test_prepared = full_pipeline.fit_transform(test_set_pre)