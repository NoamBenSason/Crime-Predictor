import numpy as np
import data_preprocessor as dpr
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from base_line import DecisionTree
from data_preprocessor import Preprocessor

TRAIN = "Dataset_crimes_train_new.csv"
VALIDATION = "Dataset_crimes_validation_new.csv"


def train_model(name, model):
    model.fit(TRAIN)
    filename = open(name + ".pkl", 'wb')
    pickle.dump(model, filename)
    filename.close()


def get_all_models(names):
    models_dict = {}
    for name in names:
        filename = open(name + ".pkl", 'rb')
        models_dict[name] = pickle.load(filename)
        filename.close()
    return models_dict



def select(models_dict, filepath):
    names = list(models_dict.keys())
    best_model = names[0]
    prep = Preprocessor()
    data, response = prep.load_data_train(filepath)
    mistakes = response.shape[0]
    print(str(mistakes))
    mistake_list = []

    for name, model in models_dict.items():
        model_response = model.predict(filepath)
        diff = (model_response != response)
        current_mistakes = np.sum(diff)
        mistake_list.append(current_mistakes)
        print(name + " : " + str(current_mistakes))
        if current_mistakes < mistakes:
            mistakes = current_mistakes
            best_model = name

    print(best_model)
    return models_dict[best_model], models_dict, mistake_list


def draw_preformance(model_list, mistake_list):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(*range(len(model_list)), mistake_list)
    plt.show()


if __name__ == "__main__":
    # TRAIN MODELS
    names = ["tree_4","tree_6", "tree_8", "tree_12","tree_14"]
    # train_model(names[0], DecisionTree(4, 0))
    train_model(names[4], DecisionTree(14, 0))
    models_dict = get_all_models(names)
    best_model, models_dict, mistake_lst = select(models_dict, TRAIN)
    #draw_preformance(names, mistake_lst)

