from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("Task2/Dataset_crimes.csv")
    train, validation = train_test_split(df, test_size=0.4, random_state=42)
    validation, test = train_test_split(validation, test_size=0.5,
                                        random_state=42)
    train.ti("Task2/Dataset_crimes_train.pkl")
    validation.to_pickle("Task2/Dataset_crimes_validation.pkl")
    test.to_pickle("Task2/Dataset_crimes_test.pkl")
    df = pd.read_pickle("Task2/Dataset_crimes_train.pkl")
    print(df.shape[0])
    print(df.head())
