import pandas as pd


def preprocess_features(features):
    features["Date"] = pd.to_datetime(features["Date"])
    features['day'] = features["Date"].dt.day
    features['month'] = features["Date"].dt.month
    features['year'] = features["Date"].dt.year
    features['time'] = features["Date"].dt.time
    features['day_of_week'] = features["Date"].dt.dayofweek
    return


def load_data():
    df = pd.read_csv("Dataset_crimes_train.csv")
    df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]

    features, response = df.drop("Primary Type", axis=1), df["Primary Type"]
    processed_features = preprocess_features(features)
    return


if __name__ == '__main__':
    load_data()
