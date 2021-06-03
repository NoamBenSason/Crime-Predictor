import pandas as pd
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE '
                                                                  'PRACTICE',
               4: 'ASSAULT'}
crimes_dict_reverse = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
                       'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def preprocess_response(response):
    process_response = response.apply(lambda x: crimes_dict_reverse.get(x))
    return process_response


def preprocess_features(features):
    features["Date"] = pd.to_datetime(features["Date"])
    features['day'] = features["Date"].dt.day
    features['month'] = features["Date"].dt.month
    features['year'] = features["Date"].dt.year
    features['time'] = features["Date"].dt.time
    features['time'] = features["time"].apply(lambda x: x.hour * 60 + x.minute)
    features['day_of_week'] = features["Date"].dt.dayofweek
    features['block_no_street'] = features["Block"].str.slice(0, 6)
    features.drop(["Block","Date"], axis=1, inplace=True)
    return features


def preprocess_all(features, response):
    return preprocess_features(features), preprocess_response(response)


def load_data(filename):
    df = pd.read_csv(filename)
    df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]
    df.dropna(inplace=True, axis=0)

    features, response = df.drop("Primary Type", axis=1), df["Primary Type"]

    return preprocess_all(features, response)


if __name__ == '__main__':
    load_data("Dataset_crimes_train.csv")
