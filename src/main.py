from generate_data2 import *
# from RNS import RNS
from lmgs import *
import pickle
import math
import os.path
import copy
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from functools import reduce
from pathlib import Path
from sklearn.metrics import confusion_matrix

###### DO NOT CHANGE ################################################################################
NO_GEN_PRIMES = 40000   # Increase if we have generated more primes                                 #
NONE = 0                                                                                            #
DATA_FOLDER = "data"                                                                                #
#####################################################################################################

###### SETTINGS #####################################################################################
BIT_LENGTH = 256        # With bit length 256, you get 87 long input, 43 moduli                     #
NO_TRAIN = 40000                                                                                    #
NO_TEST = 1000                                                                                      #
                                                                                                    #
WITHOUT_ZERO = False    # !!! Use the correct folder to separate between w/ zero and w/0 zero       #
NO_MODS = NONE          # Value NONE: no limit                                                      #
MAKE_POLY = NONE        # Value NONE: no added polynomial complexity.                               #
                        # Polynominials will only be made when #features is limited.                #
LIM_MODELS = False      # If true, we only build #NO_MODS models instead of all                     #
DATA_SUBFOLDER = f"{DATA_FOLDER}/with_zero_original"                                                #
#####################################################################################################

###### DO NOT CHANGE ################################################################################
STATS_FOLDER = f"{DATA_SUBFOLDER}/stats_{BIT_LENGTH}_#{NO_TRAIN}_#{NO_TEST}_{NO_MODS}f{MAKE_POLY}p" #
TRAIN_FILE = f"{DATA_SUBFOLDER}/train_data_{BIT_LENGTH}_#{NO_TRAIN}.p"                              #
TEST_FILE = f"{DATA_SUBFOLDER}/test_data_{BIT_LENGTH}_#{NO_TEST}.p"                                 #
MODEL_FILE = f"{DATA_SUBFOLDER}/models_{BIT_LENGTH}_#{NO_TRAIN}_{NO_MODS}f{MAKE_POLY}p.p"           #
MODEL_WEIGHTS_FILE = f"{DATA_SUBFOLDER}/models_{BIT_LENGTH}_#{NO_TRAIN}_{NO_MODS}f{MAKE_POLY}p.txt" #
PRIME_FILE = f"{DATA_FOLDER}/train_primes_#{NO_GEN_PRIMES}.p"                                       #
#####################################################################################################

def train():
    # train_data is a dictionary with as key the specific feature j (the RNS modulo) and as value 
    # a list of Datapairs (representing every inputs' prime factor in the corresponding feature, while its semiprime is represented in every feature still)
    print(MODEL_FILE)
    try:
        models = pickle.load(open(MODEL_FILE, "rb"))
    except:
        print("Models not found.")
        try:
            train_data = pickle.load(open(TRAIN_FILE, "rb"))
        except:
            if NO_TRAIN <= NO_GEN_PRIMES:
                print("Train data not found. Retrieving from prime list.")
                semiprimes = readSemiprimes()[:NO_TRAIN]
            else:
                print("Train data not found. Generating primes, this will take some time.")
                semiprimes = None
            train_data = GeneratedData(BIT_LENGTH, NO_TRAIN, semiprimes).datapairs
            pickle.dump(train_data, open(TRAIN_FILE, "wb"))
        if NO_MODS != NONE:
            train_data = selectFeatures(train_data, NO_MODS)
        print("Building models. This will take long.")
        if LIM_MODELS and NO_MODS != NONE:
            moduli = list(train_data)[:NO_MODS]
        else:
            moduli = list(train_data)
        models = {j : LMGS(train_data=train_data[j]) for j in moduli}
        pickle.dump(models, open(MODEL_FILE, "wb"))
    print("Training done.\n")
    return models

def test(models):
    # Generate test semiprimes
    try:
        gd_test = pickle.load(open(TEST_FILE, "rb"))
    except:
        gd_test = GeneratedData(BIT_LENGTH, NO_TEST)
        pickle.dump(gd_test, open(TEST_FILE, "wb"))
    test_data = gd_test.datapairs
    if NO_MODS != NONE:
        test_data = selectFeatures(test_data, NO_MODS)
    moduli = list(models)
    inputs = list(test_data.values())[0]
    no_moduli = len(moduli)
    no_features = len(inputs[0].input)

    # End statistics
    allRanks = []
    allLiks = []
    allResLiks = []     # row is entry, col is residue class
    resUniformLik = np.asarray([np.log(1/(j-1)) for j in moduli])
    uniformLik = sum(resUniformLik)
    allNormLiks = {j : [] for j in moduli}    # key residues, value list of norm_likelihoods
    actResidues = {j : np.zeros(len(inputs), dtype=int) for j in moduli} # np.zeros(j-int(WITHOUT_ZERO)
    predResidues = {j : np.zeros(len(inputs), dtype=int) for j in moduli}

    # Predict prime factors
    for index, i in enumerate(inputs):
        bools = []
        resLikLog = []
        ranks = []
        # Per modulo, predict the correct residue
        for j, model in models.items():
            possible_outputs = {res : list(GeneratedData.cos_sin(res, j)) for res in range(int(WITHOUT_ZERO),j)}
            likelihoods = {res : model.likelihood(DataPair(i.input, out, i.input_OG, res)) for res, out in possible_outputs.items()}
            norm_likelihoods = {res : lik / sum(likelihoods.values()) for res, lik in likelihoods.items()}
            
            actual = i.output_OG % j
            predicted = max(norm_likelihoods, key=norm_likelihoods.get)
            actResidues[j][index] = actual #-int(WITHOUT_ZERO)
            predResidues[j][index] = predicted #-int(WITHOUT_ZERO)
            # actResidues[j][actual-int(WITHOUT_ZERO)] += 1
            # predResidues[j][predicted-int(WITHOUT_ZERO)] += 1
            
            rankActual = sorted(norm_likelihoods, key=norm_likelihoods.get, reverse=True).index(actual) / (j-1-int(WITHOUT_ZERO))
            ranks.append(rankActual)
            # print(norm_likelihoods.get(actual))
            
            allNormLiks[j].append(list(norm_likelihoods.values()))
            # print(f"most likely res = {max3(norm_likelihoods)}\nactual res = {actual}")
            # actualIsAboveAvg = norm_likelihoods.get(actual) > 1/(j-1)
            # print(actualIsAboveAvg)
            # bools.append(actualIsAboveAvg)
            # predictedIsCloseToActual = findPeriodicDist(actual, predicted, j) <= 3 #<= math.ceil((j-1) / 3) / 2
            # print(predictedIsCloseToActual)
            # bools.append(predictedIsCloseToActual)
            resLikLog.append(np.log(norm_likelihoods.get(actual)))
        
        totalLikLog = sum(resLikLog)
        # print(f"{bools}\n#True = {bools.count(True)}\n#False = {bools.count(False)}\n{ranks}")
        # print(f"resLikLog = {resLikLog}\t totalLikLog = {totalLikLog}")  # -175.73274660169264 uniform
        allRanks.append(ranks)
        allLiks.append(totalLikLog)
        allResLiks.append(resLikLog)

    # Show average norm_likelihoods per residue class
    Path(STATS_FOLDER).mkdir(exist_ok=True)
    for j in moduli:
        plt.title(f"Normalized likelihoods for residue class {j}. Averaged over {NO_TEST} runs")
        plt.bar(tuple(range(int(WITHOUT_ZERO),j)), np.average(allNormLiks[j], axis=0))
        plt.axhline(y=1/(j-int(WITHOUT_ZERO)), linewidth=1, color='r')
        axes = plt.axes()
        if j < 7:
            axes.set_xticks(list(range(int(WITHOUT_ZERO),j)))
        # axes.set_ylim([0, 1])
        # plt.show()
        plt.xlabel("Residue")
        plt.ylabel("Normalized likelihood")
        plt.savefig(f"{STATS_FOLDER}/normalized likelihoods ({j} of {moduli[-1]}).png")
        plt.close()
    
    # Show ranks of the actual residue
    allRanks = np.asarray(allRanks)
    avgRanks = np.average(allRanks, axis=0)
    plt.bar(moduli, avgRanks)
    plt.xlabel("Residue class")
    plt.ylabel("Normalized rank of the actual residue (0.0 = most likely residue)")
    plt.title(f"Likelihood ranking of the actual residue, per residue class.\nAveraged over {NO_TEST} runs.")
    axes = plt.axes()
    axes.set_ylim([0, 1])
    plt.axhline(y=0.5, linewidth=1, color='r')
    # plt.show()
    plt.savefig(f"{STATS_FOLDER}/normalized ranks per residue.png")
    print()
    print(f"--- over {NO_TEST} runs ---")
    print(f"average log-likelihood, per residue class: {np.average(allResLiks, axis=0)}")
    print(f"uniform log-likelihood, per residue class: {resUniformLik}")
    print(f"average log-likelihood: {np.average(allLiks)}")
    print(f"uniform log-likelihood: {uniformLik}")

    # Make confusion matrices
    confusionMatrices = {j:confusion_matrix(actResidues[j], predResidues[j], list(range(int(WITHOUT_ZERO), j))) for j in moduli}
    
    # Write a summary
    with open(f"{STATS_FOLDER}/summary.txt", "w") as f:
        f.write(f"BIT_LENGTH = {BIT_LENGTH}\nNO_TRAIN = {NO_TRAIN}\nNO_TEST = {NO_TEST}\nNO_MODS = {NO_MODS}\nMAKE_POLY = {MAKE_POLY}\nLIM_MODELS = {LIM_MODELS}\nmoduli = {moduli}\nno_moduli = {no_moduli}\nno_features = {no_features}\n")
        f.write(f"\n--- over {NO_TEST} runs ---\n")
        f.write(f"average log-likelihood, per residue class: {np.average(allResLiks, axis=0)}\n")
        f.write(f"uniform log-likelihood, per residue class: {resUniformLik}\n")
        f.write(f"average log-likelihood: {np.average(allLiks)}\n")
        f.write(f"uniform log-likelihood: {uniformLik}\n")
        f.write("\n--- Confusion matrices ---\n")
        for j in moduli:
            f.write(f"\nresidue class {j}\n")
            for row in confusionMatrices[j]:
                f.write('\t'.join([str(cell) for cell in row]))
                f.write('\n')
    print("Testing done.")

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

def readSemiprimes():
    try:
        semiprimes = pickle.load(open(PRIME_FILE, "rb"))
    except:
        # For one-time use
        train_data = pickle.load(open(f"data/with_zero_original/train_data_{BIT_LENGTH}_#40000.p", "rb"))
        semiprimes = [(x.input_OG, x.output_OG) for x in train_data[5]] # take from a random key
        pickle.dump(semiprimes, open(PRIME_FILE, "wb"))
    return semiprimes

# For one-time use
def convertToWithoutZero(semiprimes):
    train_data = GeneratedData(BIT_LENGTH, NO_TRAIN, semiprimes).datapairs
    pickle.dump(train_data, open(f"data/without_zero/train_data_{BIT_LENGTH}_#40000.p", "wb"))

def selectFeatures(train_data, num):
    '''Limit our training data's input fields to only the bias and #num features'''
    return {j : [limitInputFeatures(dp, num) for dp in datapairs] for j,datapairs in train_data.items()}

def limitInputFeatures(dp, num):
    '''Limit the input list to only the bias and #num features'''
    res = [dp.input[0]]
    for i in range(num):
        res.append(dp.input[i + 1])
    mid = math.floor(len(dp.input) / 2)
    for i in range(num):
        res.append(dp.input[mid + i + 1])
    
    # Make polynomial terms
    if MAKE_POLY != NONE:
        temp = res.copy()
        res = [reduce((lambda a,b: a*b), combi) for combi in combinations_with_replacement(temp, MAKE_POLY)]
    
    return DataPair(res, dp.output, dp.input_OG, dp.output_OG)

def writeModelData(models):
    with open(MODEL_WEIGHTS_FILE, "w") as f:
        moduli = list(models)
        len_wx = len(moduli) * 2 + 1
        weightMatrix = np.zeros((len(moduli) + 1, len_wx * 2 + 1 + 1), dtype='O')
        weightMatrix[0] = ["model ->"] + [ f"w^1_{i}" for i in range(0, len_wx) ] + [ f"w^2_{i}" for i in range(0, len_wx) ] + ["sigma"]
        for i,(j,model) in enumerate(models.items()):
            weightMatrix[i+1] = [j] + model.w0 + model.w1 + [model.sigma]
        weightMatrix = np.flip(np.rot90(weightMatrix,3),1)
        for row in weightMatrix:
            f.write(f"{sepTab(row)}\n")
        # for j,model in models.items():
        #     f.write(f"{j}\nw0\n{sepTab(model.w0)}\n\nw1\n{sepTab(model.w1)}\n\nsigma\n{model.sigma}\n\n")

def sepTab(aList):
    return '\t'.join([str(x) for x in aList])

if __name__ == "__main__":
    # semiprimes = readSemiprimes()
    # convertToWithoutZero(semiprimes)
    models = train()
    # test(models)
    writeModelData(models)