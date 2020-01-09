from generate_data2 import *
# from RNS import RNS
from lmgs import *
import pickle
import math
import os.path
import copy
import matplotlib.pyplot as plt

BIT_LENGTH = 256 #192 #128  with bit length 256, you get 87 long input, 43 moduli
NO_TRAIN = 1001 #0
NO_TEST = 51 #1000
TRAIN_FILE = f"train_data_{BIT_LENGTH}_#{NO_TRAIN}.p"
MODEL_FILE = f"models_{BIT_LENGTH}_#{NO_TRAIN}.p"
TEST_FILE = f"test_data_{BIT_LENGTH}_#{NO_TEST}.p"

def train():
    # train_data is a dictionary with as key the specific feature j (the RNS modulo) and as value 
    # a list of Datapairs (representing every inputs' prime factor in the corresponding feature)
    try:
        models = pickle.load(open(MODEL_FILE, "rb"))
    except:
        # Couldn't find model, so build it from train data
        try:
            train_data = pickle.load(open(TRAIN_FILE, "rb"))
        except:
            # Couldn't find train data, so generate it
            train_data = GeneratedData(BIT_LENGTH, NO_TRAIN).datapairs
            pickle.dump(train_data, open(TRAIN_FILE, "wb"))
        models = {j : LMGS(train_data=train_data[j]) for j in list(train_data)}
        pickle.dump(models, open(MODEL_FILE, "wb"))
    return models

def test(models):
    # Generate test semiprimes
    try:
        gd_test = pickle.load(open(TEST_FILE, "rb"))
    except:
        gd_test = GeneratedData(BIT_LENGTH, NO_TEST)
        pickle.dump(gd_test, open(TEST_FILE, "wb"))
    test_data = gd_test.datapairs
    moduli = gd_test.moduli
    inputs = list(test_data.values())[0]
    # allRanks = np.zeros((inputs.length, moduli.length))

    # End statistics
    allRanks = []
    allLiks = []
    allResLiks = []     # row is entry, col is residue class
    resUniformLik = np.asarray([np.log(1/(j-1)) for j in moduli])
    uniformLik = sum(resUniformLik)
    allNormLiks = {j : [] for j in moduli}    # key residues, value list of norm_likelihoods

    for i in inputs:
        # print(i)
        bools = []
        resLikLog = []
        ranks = []
        # Per modulo, predict the correct residue
        for j, model in models.items():
            # print(f"using model {j}")
            # possible_outputs = {res : [math.cos(2*math.pi*res/j), math.sin(2*math.pi*res/j)] for res in range(1,j)}
            possible_outputs = {res : list(GeneratedData.cos_sin(res, j)) for res in range(1,j)}
            likelihoods = {res : model.likelihood(DataPair(i.input, out, i.input_OG, res)) for res, out in possible_outputs.items()}
            norm_likelihoods = {res : lik / sum(likelihoods.values()) for res, lik in likelihoods.items()}
            # print(norm_likelihoods)
            # print(f"sum = {sum(norm_likelihoods)}")
            allNormLiks[j].append(list(norm_likelihoods.values()))

            actual = i.output_OG % j
            predicted = max(norm_likelihoods, key=norm_likelihoods.get)
            # print(f"most likely res = {max3(norm_likelihoods)}\nactual res = {actual}")
            # actualIsAboveAvg = norm_likelihoods.get(actual) > 1/(j-1)
            # print(actualIsAboveAvg)
            # bools.append(actualIsAboveAvg)

            predictedIsCloseToActual = findPeriodicDist(actual, predicted, j) <= 3 #<= math.ceil((j-1) / 3) / 2
            # print(predictedIsCloseToActual)
            bools.append(predictedIsCloseToActual)
            
            rankActual = sorted(norm_likelihoods, key=norm_likelihoods.get, reverse=True).index(actual) / (j-2)
            # print(f"rank = {rankActual}")
            ranks.append(rankActual)
            
            # print(norm_likelihoods.get(actual))
            resLikLog.append(np.log(norm_likelihoods.get(actual)))
            # print()
        
        totalLikLog = sum(resLikLog)
        # print(f"{bools}\n#True = {bools.count(True)}\n#False = {bools.count(False)}\n{ranks}")
        # print(f"resLikLog = {resLikLog}\t totalLikLog = {totalLikLog}")  # -175.73274660169264 uniform
        allRanks.append(ranks)
        allLiks.append(totalLikLog)
        allResLiks.append(resLikLog)

    # Show average norm_likelihoods per residue class
    # for j in moduli:
    #     plt.title(f"Normalized likelihoods for residue class {j} averaged over {NO_TEST} runs")
    #     plt.bar(tuple(range(1,j)), np.average(allNormLiks[j], axis=0))
    #     plt.axhline(y=1/(j-1),linewidth=1, color='r')
    #     axes = plt.axes()
    #     # axes.set_ylim([0, 1])
    #     plt.show()
    #     plt.close()
    
    allRanks = np.asarray(allRanks)
    avgRanks = np.average(allRanks, axis=0)
    plt.bar(moduli, avgRanks)
    plt.xlabel("Moduli")
    plt.ylabel("Normalized rank of the actual residue (0.0 = most likely residue)")
    plt.title(f"Ranking of the actual residue, per modulo. Averaged over {NO_TEST} runs.")
    axes = plt.axes()
    axes.set_ylim([0, 1])
    plt.axhline(y=0.5,linewidth=1, color='r')
    plt.show()
    print()
    print(f"--- over {NO_TEST} runs ---")
    print(f"average log-likelihood, per residue class: {np.average(allResLiks, axis=0)}")
    print(f"uniform log-likelihood, per residue class: {resUniformLik}")
    print(f"average log-likelihood: {np.average(allLiks)}")
    print(f"uniform log-likelihood: {uniformLik}")


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