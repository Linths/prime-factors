from generate_data2 import *
# from RNS import RNS
from lmgs import *
import pickle
import math
import os.path
import copy

BIT_LENGTH = 256
NO_TRAIN = 1000
NO_TEST = 10
TRAIN_FILE = "train_data.p"
MODEL_FILE = "models.p"

def train():
    # if os.path.isfile(TRAIN_FILE):
    #     print ("File exist")
    # else:
    #     print ("File not exist")
    try:
        train_data = pickle.load(open(TRAIN_FILE, "rb"))
    except:
        train_data = GeneratedData(BIT_LENGTH, NO_TRAIN).datapairs
        pickle.dump(train_data, open(TRAIN_FILE, "wb"))
    # train_data is a dictionary with as key the specific feature j (the RNS modulo) and as value
    # a list of Datapairs (representing every inputs' prime factor in the corresponding feature)
    print(train_data.keys())
    # models = {j : LMGS(train_data=train_data[j]) for j in train_data.keys()}
    try:
        models = pickle.load(open(MODEL_FILE, "rb"))
    except:
        models = {j : LMGS(train_data=train_data[j]) for j in list(train_data)}
        pickle.dump(models, open(MODEL_FILE, "wb"))
    return models

def test(models):
    gd_test = GeneratedData(BIT_LENGTH, NO_TEST)
    test_data = gd_test.datapairs
    moduli = gd_test.moduli
    inputs = list(test_data.values())[0]
    first_inp = inputs[0]
    print(first_inp)
    for j, model in models.items():
        print(f"using model {j}")
        possible_outputs = {res : [math.cos(2*math.pi*res/j), math.sin(2*math.pi*res/j)] for res in range(1,j)}
        likelihoods = {res : model.likelihood(DataPair(first_inp.input, out, first_inp.input_OG, res)) for res, out in possible_outputs.items()}
        print(likelihoods)
        print(f"most likely res = {max3(likelihoods)}\nactual res = {first_inp.output_OG % j}")
        print()

def max3(dictio):
    dic = copy.deepcopy(dictio)
    result = []
    for _ in range(min(3, len(dictio))):
        highest = max(dic, key=dic.get)
        result.append(highest)
        dic.pop(highest)
    return result

if __name__ == "__main__":
    models = train()
    test(models)