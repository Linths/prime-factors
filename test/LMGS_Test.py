from LMGS import LMGS, DataPair
import unittest
import numpy as np

class LMGS_Test(unittest.TestCase):
    NUM_WEIGHTS = 3
    NUM_DATA_PAIRS = 10000
    RANGE = 100.0

    def test_LMGS(self):
        w0 = [np.random.uniform(-self.RANGE, self.RANGE) for _ in range(self.NUM_WEIGHTS)]
        w1 = [np.random.uniform(-self.RANGE, self.RANGE) for _ in range(self.NUM_WEIGHTS)]
        sigma = 1
        distribution = LMGS(w0, w1, sigma)
        data_pairs = []
        for _ in range(self.NUM_DATA_PAIRS):
            input = [np.random.uniform(-self.RANGE, self.RANGE) for _ in range(self.NUM_WEIGHTS)]
            rnd = distribution.generate(input)
            data_pairs.append(DataPair(input, rnd))
        trained = LMGS(train_data = data_pairs)
        print("Expected Values")
        print(w0)
        print(w1)
        print(sigma)

        print("Actual Values")
        print(trained.w0)
        print(trained.w1)
        print(trained.sigma)