import numpy as np
import data_preprocessor as dpr
import pickle
TRAIN = ""
VALIDATION = ""

def train_model(name, model):
    model.fit(TRAIN)
    filename = open(name+".pkl", 'wb')
    pickle.dump(model,  filename)
    filename.close()

def get_all_models(names):
    models_dict = {}
    for name in names:
        filename = open(name+".pkl", 'rb')
        models_dict[name] = pickle.load(filename)
        filename.close()
    return models_dict

def select(models_dict):
    names = models_dict.keys()
    best_model = names[0]
    validation_data, validation_response = dpr.load_data(VALIDATION)
    mistakes = validation_response.shape(0)
    for name, model in models_dict.items():
        model_response = model.predict(VALIDATION)
        diff = (model_response != validation_response)
        current_mistakes = np.sum(diff)
        print(name + " : " + mistakes)
        if current_mistakes < mistakes:
            mistakes = current_mistakes
            best_model = name

    print(best_model)
    return models_dict[best_model]



if __name__ == "__main__":
    #TRAIN MODELS
    names = []
    models_dict = get_all_models(names)
    best_model = select(models_dict)