from collections import namedtuple
from random import random
import numpy as np

# A Data Pair is a combination of output labels that belongs to given input values. Both input and output values are
# represented by a list of float values.
DataPair = namedtuple("DataPair", ["input", "output"])

class LMGS:
    """ LMGS is the abbreviation of Linear Multivariate Gaussian System. An LMGS assumes that data behaves like a 2D
    multivariate normal distribution, where the mean is a linear combination of the input data. And the covariance is a
    spherical, i.e. only the diagonal of the covariance matrix is filled with one value and we assume that this value
    is a given constant. """

    def __init__(self, w0: list = None, w1: list = None, sigma: float = None, train_data: list = None):
        if train_data is not None:
            w0, w1, sigma = self.__train(train_data)
        self.w0 = w0
        self.w1 = w1
        self.sigma = sigma

    def generate(self, input: list) -> list:
        mu0 = np.dot(np.array(self.w0), np.array(input))
        mu1 = np.dot(np.array(self.w1), np.array(input))
        mu = np.array([mu0, mu1])
        cov = np.array([[self.sigma, 0], [0, self.sigma]])
        return list(np.random.multivariate_normal(mu, cov, 1)[0])

    def __train(self, train_data: list) -> tuple:
        w0 = self.__get_linear_coefs(train_data, 0)
        w1 = self.__get_linear_coefs(train_data, 1)
        sigma = self.__get_sigma(train_data, w0, w1)
        return (w0, w1, sigma)

    def __get_linear_coefs(self, train_data: list, output_index: int) -> list:
        matrix = []
        equals = []
        for j in range(len(train_data[0].input)):
            equal = sum([train_data[i].input[j] * train_data[i].output[output_index] for i in range(len(train_data))])
            row = []
            for k in range(len(train_data[0].input)):
                row.append(sum([train_data[i].input[j] * train_data[i].input[k] for i in range(len(train_data))]))
            matrix.append(row)
            equals.append(equal)

        matrix = np.array(matrix)
        equals = np.array(equals)
        return list(np.linalg.solve(matrix, equals))

    def __get_sigma(self, train_data: list, w0: list, w1: list) -> float:
        sigma = 0
        for i in range(len(train_data)):
            sigma += self.__get_sigma_part(train_data, [w0, w1], i, 0)
            sigma += self.__get_sigma_part(train_data, [w0, w1], i, 1)
        return sigma / (2 * len(train_data))

    def __get_sigma_part(self, train_data: list, w: list, i: int, output_index: int) -> float:
        mu = sum([w[output_index][j] * train_data[i].input[j] for j in range(len(train_data[0].input))])
        return (train_data[i].output[output_index] - mu) ** 2