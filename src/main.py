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
    
    bools = []
    totalLikLog = 0
    uniformLikLog = 0
    # Per modulo, predict the correct residue
    for j, model in models.items():
        print(f"using model {j}")
        possible_outputs = {res : [math.cos(2*math.pi*res/j), math.sin(2*math.pi*res/j)] for res in range(1,j)}
        likelihoods = {res : model.likelihood(DataPair(first_inp.input, out, first_inp.input_OG, res)) for res, out in possible_outputs.items()}
        norm_likelihoods = {res : lik / sum(likelihoods) for res, lik in likelihoods.items()}
        print(norm_likelihoods)
        actual = first_inp.output_OG % j
        predicted = max(norm_likelihoods, key=norm_likelihoods.get)
        print(f"most likely res = {max3(norm_likelihoods)}\nactual res = {actual}")
        actualIsAboveAvg = norm_likelihoods.get(actual) > 1/(j-1)
        print(actualIsAboveAvg)
        bools.append(actualIsAboveAvg)
        # predictedIsCloseToActual = findPeriodicDist(actual, predicted, j) <= math.ceil((j-1) / 3)
        # print(predictedIsCloseToActual)
        # bools.append(predictedIsCloseToActual)
        print(norm_likelihoods.get(actual))
        totalLikLog += np.log(norm_likelihoods.get(actual))
        uniformLikLog += np.log(1/(j-1))
        print()
    
    print(f"{bools}\n#True = {bools.count(True)}\n#False = {bools.count(False)}")
    print(f"totalLikLog = {totalLikLog}\nuniformLikLog = {uniformLikLog}")


def findPeriodicDist(n1, n2, j):
    '''Find the distance between n1 and n2, where numbers are spread from 1..(j-1) and where 1 and j-1 are neighbours. (It wraps and we skip 0!)
    Assumes n1 != j, n1 != 0, n2 != j, n2 != 0 '''
    # Ensure n2 >= n1
    if n2 < n1:
        n1, n2 = n2, n1
    # Find the distance without using 'wrapping'
    dist1 = n2 - n1
    # Find the distance when 'wrapping' (passing j-1 and 1)
    dist2 = (j-1) - n2 + n1 
    return min(dist1, dist2)


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
    test(models)