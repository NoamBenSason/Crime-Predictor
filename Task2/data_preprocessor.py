import pandas as pd

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE '
                                                                  'PRACTICE',
               4: 'ASSAULT'}
crimes_dict_reverse = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
                       'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def preprocess_features(features):
    features["Date"] = pd.to_datetime(features["Date"])
    features['day'] = features["Date"].dt.day
    features['month'] = features["Date"].dt.month
    features['year'] = features["Date"].dt.year
    features['time'] = features["Date"].dt.time
    features['day_of_week'] = features["Date"].dt.dayofweek
    return


def load_data(filename):
    df = pd.read_csv(filename)
    df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]

    features, response = df.drop("Primary Type", axis=1), df["Primary Type"]
    features.dropna(inplace=True,axis=0)
    processed_features = preprocess_features(features)

    return


if __name__ == '__main__':
    load_data("Dataset_crimes_test.csv")
