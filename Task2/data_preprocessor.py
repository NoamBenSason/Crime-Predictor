import pandas as pd


def preprocess_features(features):
    pass


def load_data():
    df = pd.read_csv("Dataset_crimes_train.csv")
    df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]

    features, response = df.drop("Primary Type",axis=1), df["Primary Type"]
    return


if __name__ == '__main__':
    load_data()





