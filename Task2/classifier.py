import pickle
import pandas as pd
from data_preprocessor import Preprocessor

MODEL = "chosen_model.pkl"
CLUSTER_MODEL = "cluster_dict.pkl"
crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def predict(X):
    filename = open(MODEL, 'rb')
    model = pickle.load(filename)
    filename.close()
    response = model.predict(X)
    return response


def send_police_cars(X):
    filename = open(CLUSTER_MODEL, 'rb')
    model = pickle.load(filename)
    filename.close()
    df = pd.DataFrame({"dates": X})
    df["dates"] = pd.to_datetime(df["dates"])
    df["dates"] = df["dates"].dt.day_name()
    df["dates"] = df["dates"].apply(lambda x: model[x])

    return df["dates"].values


if __name__ == '__main__':
    print(predict("Dataset_crimes_with_new.csv"))
