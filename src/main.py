from generate_data2 import *
# from RNS import RNS
from lmgs import *
import pickle
import math
import os.path

BIT_LENGTH = 256
NO_TRAIN = 1000
NO_TEST = 10
TRAIN_FILE = "train_data.p"
models = {}

def train():
    if os.path.isfile(TRAIN_FILE):
        print ("File exist")
    else:
        print ("File not exist")
    try:
        train_data = pickle.load(open(TRAIN_FILE, "rb"))
    except:
        train_data = generate_data(BIT_LENGTH, NO_TRAIN)
        pickle.dump(train_data, open(TRAIN_FILE, "wb"))
    # train_data is a dictionary with as key the specific feature j (the RNS modulo) and as value
    # a list of Datapairs (representing every inputs' prime factor in the corresponding feature)
    models = {j : LMGS(datapairs) for j,datapairs in train_data.items()}

def test():
    test_data = generate_data(BIT_LENGTH, NO_TEST)
    first_test = test_data[0]
    print(first_test)
    moduli = find_firsts()
    for j, model in models.items():
        print(f"using model {j}")
        possible_outputs = {res : [math.cos(2*math.pi*res/j), math.sin(2*math.pi*res/j)] for res in range(1,j)}
        likelihoods = {res : model.likelihood(DataPair(first_test[0], out)) for res, out in possible_outputs}
        print(likelihoods)
        print(f"most likely res = {max(likelihoods, key=likelihoods.get)}")

if __name__ == "__main__":
    train()
    test()