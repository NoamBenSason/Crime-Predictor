import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance_matrix
import datetime as dt
import numpy as np
import data_preprocessor as dpr
import pickle

TRAIN_PATH = "Dataset_crimes_with_new.csv"

fit_per_day_dict = {'Sunday': [], 'Monday': [], 'Tuesday': [], 'Wednesday': [],
                    'Thursday': [], 'Friday': [],
                    'Saturday': []}

CLUSTERS_NUM = 30


def calc_affinity_matrix(X):
    # calculating the affinity matrix
    A = distance_matrix(X, X)
    A = np.exp(-A / 7751)

    return A


def affinity_rbf(X):
    # calculating the affinity matrix
    A = calc_affinity_matrix(X)

    # Building the clustering model
    spectral_model_rbf = SpectralClustering(n_clusters=CLUSTERS_NUM,
                                            affinity='precomputed')
    # Training the model and Storing the predicted cluster labels
    labels_rbf = spectral_model_rbf.fit_predict(A)

    return labels_rbf


def affinity_knn(X):
    # Building the clustering model
    spectral_model_nn = SpectralClustering(n_clusters=CLUSTERS_NUM,
                                           affinity='nearest_neighbors')

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

def return_time_to_normal(time):
    time = int(str(time)[:3])
    hour = int(time // 60)
    minute = time % 60
    return dt.time(hour=hour, minute=minute, second=00)


def fit():
    proc = dpr.Preprocessor()
    X = proc.load_data_second(TRAIN_PATH)

    for day in fit_per_day_dict.keys():

        X_per_day = X[X['Day Of Week'] == day]
        X_per_day = X_per_day.drop(['Day Of Week'], axis=1)
        labels = affinity_rbf(X_per_day)

        centroids = []

        for i in range(CLUSTERS_NUM):
            vec = X_per_day[labels == i].mean()
            centroids.append(tuple(
                [np.round(vec["X Coordinate"]).astype(int), np.round(vec["Y Coordinate"]).astype(int),
                 return_time_to_normal(vec["time"])]))  # tuple definition
        fit_per_day_dict[day] = centroids

        # df_labels = pd.DataFrame(labels, columns=["cluster"])
        # X_per_day.reset_index(drop=True, inplace = True)
        # df_concat = pd.concat([X_per_day, df_labels], axis=1, join="inner")



def send_police_cars(dates):
    df = pd.DataFrame({"dates": dates})
    df["dates"] = pd.to_datetime(df["dates"])
    df["dates"] = df["dates"].dt.day_name()
    df["dates"] = df["dates"].apply(lambda x: fit_per_day_dict[x])

    return df["dates"].values


if __name__ == '__main__':
    fit()
    filename = open("cluster_dict.pkl", 'wb')
    pickle.dump(fit_per_day_dict, filename)
    filename.close()

    # filename = open("cluster_dict.pkl", 'rb')
    # fit_per_day_dict = pickle.load(filename)
    #
    # for key in fit_per_day_dict.keys():
    #     print("day: " + str(key) + " 30 values: ", str(fit_per_day_dict[key]))
    #
    # dates = ["06/03/2021 01:23:00 PM", "05/03/2021 01:23:00 PM"]
    # print(send_police_cars(dates))