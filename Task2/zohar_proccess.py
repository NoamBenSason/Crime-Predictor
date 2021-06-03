import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE '
                  'PRACTICE',
               4: 'ASSAULT'}
crimes_dict_reverse = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
                       'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def preprocess_response(response):
    process_response = response.apply(lambda x:
                                      crimes_dict_reverse.get(
                                          x))
    return process_response



def preprocess_all(features, response):
    return preprocess_features(features), preprocess_response(
        response)



def preprocess_features(features):
    features["Date"] = pd.to_datetime(features["Date"])
    features['day'] = features["Date"].dt.day
    features['month'] = features["Date"].dt.month
    features['year'] = features["Date"].dt.year
    features['time'] = features["Date"].dt.time
    features['day_of_week'] = features["Date"].dt.dayofweek
    features[features['Arrest'] != "TRUE" or features['Arrest'] != "FALSE"] = False
    features[features['Domestic'] != "TRUE" or features['Arrest'] != "FALSE"] = False
    features[features.applymap(np.isreal)['Beat'] == False] = -1
    features[features.applymap(np.isreal)['District'] == False] = -1
    features[features.applymap(np.isreal)['Ward'] == False] = -1
    features[features.applymap(np.isreal)['Community Area'] == False] = -1
    features[features.applymap(np.isnan)['Beat']] = -1
    features[features.applymap(np.isnan)['District']] = -1
    features[features.applymap(np.isnan)['Ward']] = -1
    features[features.applymap(np.isnan)['Community Area']] = -1
    meanX = features[features.applymap(np.isreal)['X Coordinate']]['X Coordinate'].mean()
    meanY = features[features.applymap(np.isreal)['Y Coordinate']]['Y Coordinate'].mean()
    features[features.applymap(np.isreal)['X Coordinate'] == False] = meanX
    features[features.applymap(np.isnan)['X Coordinate']] = meanX
    features[features.applymap(np.isreal)['Y Coordinate'] == False] = meanY
    features[features.applymap(np.isnan)['Y Coordinate']] = meanY
    return

def load_data_train(filename):
    df = pd.read_csv(filename)
    df = df[
        ["Date", "Block", "Primary Type", "Location Description", "Arrest",
         "Domestic", "Beat", "District", "Ward", "Community Area",
         "X Coordinate", "Y Coordinate"]]
    features, response = df.drop("Primary Type", axis=1), df["Primary Type"]

    return preprocess_all(features, response)


def load_data(filename):
    df = pd.read_csv(filename)
    df = df[
        ["Date", "Block", "Primary Type", "Location Description", "Arrest",
         "Domestic", "Beat", "District", "Ward", "Community Area",
         "X Coordinate", "Y Coordinate"]]
    features = df

    return preprocess_features(features)





if __name__ == '__main__':
    load_data_train("Dataset_crimes_train.csv")
