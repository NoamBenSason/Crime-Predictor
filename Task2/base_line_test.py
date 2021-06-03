import base_line

TRAIN_PATH = "Dataset_crimes_train_new.csv"
VALIDATION_PATH = "Dataset_crimes_validation_new.csv"


def main():
    dt = base_line.DecisionTree(4)
    dt.fit(TRAIN_PATH)
    dt.predict(VALIDATION_PATH)


if __name__ == '__main__':
    main()
