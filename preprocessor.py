from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("Task2/Dataset_crimes.csv")
    df2 = pd.read_csv("Task2/crimes_dataset_part2.csv")

    co = pd.concat([df, df2], axis=0)
    co.to_csv("Task2/Dataset_crimes_with_new.csv")

    train, validation = train_test_split(co, test_size=0.4, random_state=42)
    validation, test = train_test_split(validation, test_size=0.5,
                                        random_state=42)
    train.to_csv("Task2/Dataset_crimes_train_new.csv")
    validation.to_csv("Task2/Dataset_crimes_validation_new.csv")
    test.to_csv("Task2/Dataset_crimes_test_new.csv")
