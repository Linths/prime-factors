import math
from collections import namedtuple
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
        """ Constructor of a Linear Multivariate Gaussian System. You should either give as parameters:
            - w0, w1 and sigma, which directly define the behavior of the model.
            - train_data which is used to train the Linear Multivariate Gaussian System (by using a closed formula for
              the maximum likelihood estimator)

        Parameters:
            w0 (list): A list of floats which are the weights that determine the mean for the first label.
            w1 (list): A list of floats which are the weights that determine the mean for the second label (this list
            should be of the same size as w0).
            sigma (float): Determines the variance in the model (the higher this value the more uncertain this model is
                about the behavior of the data). Sigma should always be strict larger than 0.
            train_data (list): A list of Data Pair which is the train data used to train the parameters w0, w1 and sigma.
                We assume that the input list has the same length for every given Data Pair. Likewise this should also
                hold for the output list for every given Data Pair.

        Raises:
            PerfectFitError: This error can be thrown when the LMGS is trained with training data. If a linear model
                perfectly fits through the training data then the program raises a Perfect Fit Error, because a
                Linear Multivariate Gaussian System does not support a sigma equal to zero.
        """
        if train_data is not None:
            w0, w1, sigma = self.__train(train_data)
        self.w0 = w0
        self.w1 = w1
        self.sigma = sigma

    def likelihood(self, data_pair: DataPair) -> float:
        """ Check the likelihood (probability density) that given the input of the data pair it produces the output of
        the data pair.

        Arguments:
            data_pair (DataPair): The data_pair of which the likelihood (probability density) will be checked.

        Returns:
            The likelihood (probability density) that given the input of the data pair it generates the output of that
            data pair.
        """
        mu0 = np.dot(np.array(self.w0), np.array(data_pair.input))
        mu1 = np.dot(np.array(self.w1), np.array(data_pair.input))
        exp = (data_pair.output[0] - mu0) ** 2 + (data_pair.output[1] - mu1) ** 2
        exp /= -2 * self.sigma
        return math.exp(exp) / (2 * math.pi * self.sigma)

    def generate(self, input: list) -> list:
        """ Generate a random label for some given input using this model.

        Parameters:
            input (list): A list of floats which should be of the same length as the w0 and w1 parameters of the model.
                For this input data we will generate randomly a label using this model.

        Returns:
            The randomly generated output label for the given input data, which is a list of floats.
        """
        mu0 = np.dot(np.array(self.w0), np.array(input))
        mu1 = np.dot(np.array(self.w1), np.array(input))
        mu = np.array([mu0, mu1])
        cov = np.array([[self.sigma, 0], [0, self.sigma]])
        return list(np.random.multivariate_normal(mu, cov, 1)[0])

    def __train(self, train_data: list) -> tuple:
        """ Compute the parameters of this model using a closed formula for the maximum likelihood estimator. """
        w0 = self.__get_linear_coefs(train_data, 0)
        w1 = self.__get_linear_coefs(train_data, 1)
        sigma = self.__get_sigma(train_data, w0, w1)
        if sigma == 0:
            raise Exception("PerfectFitError", "Train Data has a perfect linear fit")
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