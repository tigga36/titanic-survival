import os
import pandas as pd

TITANIC_PATH = os.path.join("raw", "titanic")

def load_titanic_data(titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path,"train.csv")
    return pd.read_csv(csv_path)

titanic = load_titanic_data()
print(titanic.head())