import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix
import datetime as dt
import numpy as np
import data_preprocessor as dpr

TRAIN_PATH = "Dataset_crimes_train_new.csv"

fit_per_day_dict = {'Sunday': [], 'Monday': [], 'Tuesday': [], 'Wednesday': [], 'Thursday': [], 'Friday': [],
                    'Saturday': []}

CLUSTERS_NUM = 30


def calc_affinity_matrix(X):
    # calculating the affinity matrix
    A = distance_matrix(X, X)
    A = np.exp(-A / 50)
    return A


def affinity_rbf(X):
    # calculating the affinity matrix
    A = calc_affinity_matrix(X)

    # Building the clustering model
    spectral_model_rbf = SpectralClustering(n_clusters=CLUSTERS_NUM, affinity='precomputed')
    # Training the model and Storing the predicted cluster labels
    labels_rbf = spectral_model_rbf.fit_predict(A)

    return labels_rbf


def affinity_knn(X):
    # Building the clustering model
    spectral_model_nn = SpectralClustering(n_clusters=CLUSTERS_NUM, affinity='nearest_neighbors')

    # Training the model and Storing the predicted cluster labels
    labels_nn = spectral_model_nn.fit_predict(X)
    return labels_nn


# def evaluate():
#     # List of different values of affinity
#     affinity = ['rbf', 'nearest-neighbours']
#
#     # List of Silhouette Scores
#     s_scores = []
#
#     # Evaluating the performance
#     s_scores.append(silhouette_score(X, labels_rbf))
#     s_scores.append(silhouette_score(X, labels_nn))
#
#     print(s_scores)
#


def fit():
    proc = dpr.Preprocessor()
    X = proc.load_data_second(TRAIN_PATH)

    for day in fit_per_day_dict.keys():

        X_per_day = X[X['Day Of Week'] == day]
        X_per_day = X_per_day.drop(['Day Of Week'], axis=1)
        labels = affinity_rbf(X_per_day)

        centroids = np.zeros((X_per_day.shape[0], X_per_day.shape[1]))

        for i in range(CLUSTERS_NUM):
            centroids[i] = X_per_day[X_per_day == i].mean()

        fit_per_day_dict[day] = centroids


def send_police_cars(dates):
    centroids_for_all_date = []

    for date in dates:
        day = pd.to_datetime(date).dt.day_name()
        centroids_for_all_date.append(tuple(fit_per_day_dict[day]))

    return centroids_for_all_date


if __name__ == '__main__':
    # print(send_police_cars('2015-01-01'))
    fit()

    for key in fit_per_day_dict.keys():
        print("day: " + str(key) + " 30 vals: ", str(fit_per_day_dict[key]))
