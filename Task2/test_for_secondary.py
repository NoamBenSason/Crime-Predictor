import secondary as sc
import pandas as pd
import plotly.express as px

VALIDATION_PATH = "Dataset_crimes_validation_new.csv"


def main():
    df = sc.fit()

    loc_fig = px.scatter_3d(df, x="X Coordinate", y="Y Coordinate", z="time", color="cluster", title="Spectral Clustering of Chicago")
    loc_fig.show()


if __name__ == '__main__':
    main()
