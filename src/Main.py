from GenerateData import GenerateData as gd
from RNS import RNS
from LMGS import *
import pickle

def train():
    bit_length = 256
    sample_size = 1000
    # data is a list of Datapairs (as many as sample_size)
    try:
        train_data = pickle.load(open("train_data.p", "rb"))
    except:
        train_data = gd.generate_data(sample_size, bit_length)
        pickle.dump(train_data, open("train_data.p", "wb"))
    rns = RNS(bit_length)
    train_data = [DataPair(rns.transform(x[0]), rns.transform(x[1])) for x in train_data]
    print(train_data)

if __name__ == "__main__":
    train()