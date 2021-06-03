import base_line

DATA_PATH = r"C:\Users\noamb\Desktop\ImlHackton\Task2\Dataset_crimes_train.csv"


def main():
    dt = base_line.DecisionTree(4)
    dt.fit(DATA_PATH)


if __name__ == '__main__':
    main()
