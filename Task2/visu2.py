import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from data_preprocessor import *

import plotly.graph_objects as go

if __name__ == '__main__':

    # df = pd.read_csv("Dataset_crimes_train.csv")
    # df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
    #          "Domestic", "Beat", "District", "Ward", "Community Area",
    #          "X Coordinate", "Y Coordinate"]]
    #
    # features, response = df.drop("Primary Type", axis=1), df["Primary Type"]

    # colors = {'CRIMINAL DAMAGE': 'red', 'ASSAULT': 'green', 'DECEPTIVE PRACTICE': 'blue', 'BATTERY': 'yellow', 'THEFT': 'black'}
    #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(df["X Coordinate"], df["Y Coordinate"], c=df["Primary Type"].map(colors), s=0.1)
    # plt.show()

    features, response = load_data("Dataset_crimes_train.csv")
    print(features.describe())
    df = pd.concat([features, response], axis=1)
    loc_fig = px.scatter(df, x="X Coordinate", y="Y Coordinate", color="Primary Type")
    loc_fig.show()

    time_fig = px.histogram(df, x="time", nbins=96)
    time_fig.show()

    time_fig_pro = px.histogram(df, x="time", nbins=24, color="Primary Type")
    time_fig_pro.show()

    time_fig_pro = px.histogram(df, x="block", nbins=24, color="Primary Type")
    time_fig_pro.show()




