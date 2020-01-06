from generate_data2 import *
# from RNS import RNS
from lmgs import *
import pickle
import math
import os.path
import copy

BIT_LENGTH = 256 #192 #128  with bit length 256, you get 87 long input, 43 moduli
NO_TRAIN = 10000 #0
NO_TEST = 10
TRAIN_FILE = f"train_data_{BIT_LENGTH}_#{NO_TRAIN}.p"
MODEL_FILE = f"models_{BIT_LENGTH}_#{NO_TRAIN}.p"

def train():
    # if os.path.isfile(TRAIN_FILE):
    #     print ("File exists")
    # else:
    #     print ("File does not exist")
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
    # Generate test semiprimes
    gd_test = GeneratedData(BIT_LENGTH, NO_TEST)
    test_data = gd_test.datapairs
    moduli = gd_test.moduli
    
    # The generated test semiprimes
    inputs = list(test_data.values())[0]
    # Limiting ourselves to just one semi-prime now
    first_inp = inputs[0]
    print(first_inp)
    
    # Per modulo, predict the correct residue
    for j, model in models.items():
        print(f"using model {j}")
        possible_outputs = {res : [math.cos(2*math.pi*res/j), math.sin(2*math.pi*res/j)] for res in range(1,j)}
        likelihoods = {res : model.likelihood(DataPair(first_inp.input, out, first_inp.input_OG, res)) for res, out in possible_outputs.items()}
        print(likelihoods)
        print(f"most likely res = {max3(likelihoods)}\nactual res = {first_inp.output_OG % j}\nactual res with sin/cos = {first_inp.output}")
        print()

def max3(dictio):
    ''' Determine the top 3 residues with the highest likelihood.
    Or more generally, for a dictionary, the top 3 keys with the highest value. '''
    dic = copy.deepcopy(dictio)
    result = []
    for _ in range(min(3, len(dictio))):
        highest = max(dic, key=dic.get)
        result.append((highest, dic.get(highest)))
        dic.pop(highest)
    return result

if __name__ == "__main__":
    models = train()
    # test(models)