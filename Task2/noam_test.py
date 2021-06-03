import base_line

DATA_PATH = r"C:\Users\noamb\Desktop\UNI\YEAR_2\IML_67577\Hackthon\my_data_test.csv"


def main():
    dt = base_line.DecisionTree(4)
    dt.fit(DATA_PATH)


if __name__ == '__main__':
    main()
