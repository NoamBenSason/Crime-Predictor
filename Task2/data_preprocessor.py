import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE '
                  'PRACTICE',
               4: 'ASSAULT'}
crimes_dict_reverse = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
                       'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


class Preprocessor:

    def __init__(self):
        self.itemsBlock = None
        self.itemsLocation = None

    def preprocess_response(self, response):
        process_response = response.apply(lambda x:
                                          crimes_dict_reverse.get(
                                              x))
        return process_response

    def preprocess_features(self, features):
        features["Date"] = pd.to_datetime(features["Date"])
        features['day'] = features["Date"].dt.day
        features['month'] = features["Date"].dt.month
        features['year'] = features["Date"].dt.year
        features['time'] = features["Date"].dt.time
        features['time'] = features["time"].apply(
            lambda x: x.hour * 60 + x.minute)
        features['day_of_week'] = features["Date"].dt.dayofweek
        features['Beat'] = features['Beat'].fillna(-1)
        features['District'] = features['District'].fillna(-1)
        features['Ward'] = features['Ward'].fillna(-1)
        features['Community Area'] = features['Community Area'].fillna(-1)
        meanX = features[features.applymap(np.isreal)['X Coordinate']][
            'X Coordinate'].mean()
        meanY = features[features.applymap(np.isreal)['Y Coordinate']][
            'Y Coordinate'].mean()
        features['X Coordinate'] = features['X Coordinate'].fillna(meanX)
        features['Y Coordinate'] = features['Y Coordinate'].fillna(meanY)
        # features['block_no_street'] = features["Block"].str.slice(0, 5)
        # if self.itemsBlock is None:
        #     self.itemsBlock = features['block_no_street'].value_counts().axes[
        #                           0].values[:85]
        # features.loc[~(features['block_no_street'].isin(self.itemsBlock))]['block_no_street'] = 'other'
        # features = pd.get_dummies(features, prefix='block', columns=[
        #     'block_no_street'])
        # if self.itemsLocation is None:
        #     self.itemsLocation = \
        #         features["Location Description"].value_counts().axes[
        #             0].values[:20]
        # features.loc[~(features["Location Description"].isin(
        #     self.itemsLocation))]["Location Description"] = 'other'
        # features = pd.get_dummies(features, prefix='location', columns=[
        #     "Location Description"])
        features['Location Description'] = features['Location Description'].fillna('')
        features['is_STORE'] = features['Location Description'].str.contains('STORE').astype(int)
        features['is_APARTMENT'] = features['Location Description'].str.contains('APARTMENT').astype(int)
        features['is_STREET'] = features['Location Description'].str.contains('STREET').astype(int)
        features['is_PARKING'] = features['Location Description'].str.contains('PARKING').astype(int)
        features['is_RESIDENCE'] = features['Location Description'].str.contains('RESIDENCE').astype(int)
        features['is_SIDEWALK'] = features['Location Description'].str.contains('SIDEWALK').astype(int)



        features.drop(["Date", "Block", "Location Description"], inplace=True,
                      axis=1)
        return features

    def preprocess_all(self, features, response):
        return self.preprocess_features(features), self.preprocess_response(
            response)

    def load_data_train(self, filename):
        df = pd.read_csv(filename)
        df = df[
            ["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]
        features, response = df.drop("Primary Type", axis=1), df["Primary Type"]

        return self.preprocess_all(features, response)

    def load_data_test(self, filename):
        df = pd.read_csv(filename)
        df = df[
            ["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]
        features = df

        return self.preprocess_features(features)

    def preprocess_features_second(self, features):
        features["Date"] = pd.to_datetime(features["Date"])
        features['day'] = features["Date"].dt.day
        features['month'] = features["Date"].dt.month
        features['year'] = features["Date"].dt.year
        features['time'] = features["Date"].dt.time
        features.drop(['Date'], inplace=True,
                      axis=1)
        return

    def load_data_second(self, filename):
        df = pd.read_csv(filename)
        features = df[["Date", "X Coordinate", "Y Coordinate"]]
        df.dropna(inplace=True, axis=0)
        return self.preprocess_features_second(features)


if __name__ == '__main__':
    preprocess = Preprocessor()
    preprocess.load_data_train("Dataset_crimes_train.csv")
