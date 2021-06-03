import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv("Dataset_crimes_train")
    df = df[["Date", "Block", "Primary Type", "Location Description", "Arrest",
             "Domestic", "Beat", "District", "Ward", "Community Area",
             "X Coordinate", "Y Coordinate"]]

    features, response = df.drop("Primary Type", axis=1), df["Primary Type"]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(features["X Coordinate"], features["Y Coordinate"], label=response["Primary Type"])
    plt.show()