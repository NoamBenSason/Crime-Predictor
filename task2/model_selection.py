import numpy as np
import pickle
from base_line import DecisionTree
from forest import RandomForest
from data_preprocessor import Preprocessor

TRAIN = "Dataset_crimes_with_new.csv"
VALIDATION = "Dataset_crimes_validation_new.csv"
TEST = "Dataset_crimes_test_new.csv"


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

def get_all_models_no_save(names, models):
    models_dict = {}
    for i in range(len(names)):
        models[i].fit(TRAIN)
        models_dict[names[i]] = models[i]
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
        curretn_missclass = np.sum(diff)/data.shape[0]
        mistake_list.append(curretn_missclass)
        print(name + " : " + str(curretn_missclass))
        if curretn_missclass < mistakes:
            mistakes = curretn_missclass
            best_model = name

    print(best_model)
    return models_dict[best_model], models_dict, mistake_list


def select_no_save(models_dict, filepath):
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
        curretn_missclass = np.sum(diff)/data.shape[0]
        mistake_list.append(curretn_missclass)
        print(name + " : " + str(curretn_missclass))
        if curretn_missclass < mistakes:
            mistakes = curretn_missclass
            best_model = name

    print(best_model)
    return models_dict[best_model], models_dict, mistake_list


# def draw_preformance(model_list, mistake_list):
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot([i+1 for i in range(len(model_list))], mistake_list)
#     ax.set_xlabel("max features")
#     ax.set_ylabel("misclassification error")
#     ax.title.set_text("first feature selection")
#     plt.show()


if __name__ == "__main__":
    # TRAIN MODELS
    names = []
    models = []

    # for i in range(1):
    #     models.append(RandomForest((16, 9, 10, 75, 0.0, 1)))
    #     names.append("random forest"+str(i+1))
    #     #train_model(names[i], RandomForest(16, i+1, 10, 100, 0.0, 1))
    #

    names = ["chosen_model"]
    train_model(names[0], RandomForest(16, 9, 10, 75, 0.0, 1))
    # models = [RandomForest(None, i, 60, 2, 0.0, 1), RandomForest(1, i, 60, 2, 0.0, 1)]
    # models_dict = get_all_models(names)
    # best_model, models_dict, missclass_lst = select(models_dict, TEST)
    #draw_preformance(names, missclass_lst)




