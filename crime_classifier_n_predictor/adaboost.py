import pandas as pd
from data_preprocessor import Preprocessor
from sklearn.ensemble import AdaBoostClassifier

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


class Adaboost:
    def __init__(self, num):
        self.adaboost = AdaBoostClassifier(n_estimators=num)

    def fit(self, train_csv_path):
        """
        Given a path to a training this method learns the
        parameters of the model and stores the trained model.
        :param train_csv_path: path to the train data
        :return: nothing
        """
        self.pre = Preprocessor()
        x_train, y_train = self.pre.load_data_train(train_csv_path)
        self.adaboost.fit(x_train, y_train)

    def predict(self, predict_csv_path):
        """
        This function receives a path to a csv file with the feature columns (as in the training set) of crimes and
        predicts for each one which crime has occurred.
        :param predict_csv_path: path to the data to predict
        :return: list (or a one dimension numpy array) of labels (ints between {0-4} for the 5 classes)
        """
        x_predict,y = self.pre.load_data_train(predict_csv_path)

        return self.adaboost.predict(x_predict)
