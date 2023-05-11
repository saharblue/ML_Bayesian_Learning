import itertools
import math

import numpy as np


class conditional_independence():

    def __init__(self):
        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.08,
            (0, 1): 0.01,
            (1, 0): 0.08,
            (1, 1): 0.01
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.17,
            (1, 0): 0.34,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.17,
            (1, 0): 0.34,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.0578,
            (0, 1, 0): 0.102,
            (0, 1, 1): 0.119,
            (1, 0, 0): 0.102,
            (1, 0, 1): 0.119,
            (1, 1, 0): 0.2312,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are dependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        cartesian_product = itertools.product([0, 1], [0, 1])
        for prod in cartesian_product:
            current_x = prod[0]
            current_y = prod[1]
            if not np.isclose(X[current_x] * Y[current_y], X_Y[prod]):
                return True
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are independent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        cartesian_product = itertools.product([0, 1], [0, 1], [0, 1])
        for prod in cartesian_product:
            current_x = prod[0]
            current_y = prod[1]
            current_c = prod[2]
            if not np.isclose(X_C[(current_x, current_c)] * Y_C[(current_y, current_c)] / C[current_c], X_Y_C[prod]):
                return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    rate = float(rate)
    k = int(k)
    log_p = (rate ** k) * (math.e ** (-1 * rate)) / math.factorial(k)
    log_p = np.log(log_p)
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))
    # calculate log likelihood for each rate
    for i, rate in enumerate(rates):
        log_likelihood = 0
        for k in samples:
            log_likelihood += poisson_log_pmf(k, rate)
        likelihoods[i] = log_likelihood
    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    max_index = np.argmax(likelihoods)
    return rates[max_index]


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    return np.mean(samples)


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.
    """
    p = (math.e ** (((x - mean) ** 2) / (-2 * (std ** 2)))) / np.sqrt(2 * math.pi * (std ** 2))
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.filtered_data = dataset[dataset[:, -1] == class_value]
        self.total_size = dataset.shape[0]
        self.class_size = self.filtered_data.shape[0]
        self.mean = np.mean(self.filtered_data[:, :-1], axis=0)
        self.std = np.std(self.filtered_data[:, :-1], axis=0)
        self.class_value = class_value

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = self.filtered_data.shape[0] / self.total_size
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = np.prod(normal_pdf(x, self.mean, self.std), axis=0)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object containing the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return self.ccd0.class_value
        else:
            return self.ccd1.class_value


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput:
        - Accuracy = #Correctly Classified / test_set size
    """
    # apply prediction function on dataset
    classified = np.apply_along_axis(map_classifier.predict, axis=1, arr=test_set[:, :-1])
    # count how many predictions were true / total_set size
    total = np.count_nonzero(np.equal(classified, test_set[:, -1])) / test_set.shape[0]
    return total


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal density function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    size = len(x)

    # Normalize x
    x_m = x - mean

    if len(x.shape) == 1:
        x_m = x_m.reshape(-1, 1)

    # Probability Density Function
    pdf = (1 / (np.sqrt((2 * math.pi) ** size * np.linalg.det(cov))) *
           np.exp(-(x_m.T.dot(np.linalg.inv(cov)).dot(x_m)) / 2))

    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.filtered_data = dataset[dataset[:, -1] == class_value]
        self.cov = np.cov(self.filtered_data[:, :-1], rowvar=False)
        self.mean = np.mean(self.filtered_data[:, :-1], axis=0)
        self.total_size = len(dataset)
        self.class_size = len(self.filtered_data)
        self.class_value = class_value

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = self.class_size / self.total_size
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = np.prod(multi_normal_pdf(x, self.mean, self.cov), axis=0)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MaxPrior():
    def __init__(self, ccd0: MultiNormalClassDistribution, ccd1: MultiNormalClassDistribution):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the prior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            return self.ccd0.class_value
        else:
            return self.ccd1.class_value


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return self.ccd0.class_value
        else:
            return self.ccd1.class_value


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        filtered_data = dataset[dataset[:, -1] == class_value]
        self.class_size = len(filtered_data)
        self.class_value = class_value

    def get_prior(self):
        """
        Returns the prior porbability of the class
        according to the dataset distribution.
        """
        return self.class_size

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
        """
        likelihood = np.prod(np.array([self.check_prob(value, index) for index, value in enumerate(x)]))
        return likelihood

    def check_prob(self, x, feature):
        """
        Returns the probability of a value of a feature
        """
        filtered_data = self.dataset[self.dataset[:, -1] == self.class_value]
        nij = filtered_data[filtered_data[:, feature] == x].shape[0]
        vj = len(np.unique(self.dataset[feature]))
        if nij == 0:
            return EPSILLON
        else:
            return (nij + 1) / (self.class_size + vj)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return self.ccd0.class_value
        else:
            return self.ccd1.class_value

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        # apply prediction function on dataset
        classified = np.apply_along_axis(self.predict, axis=1, arr=test_set[:, :-1])
        # count how many predictions were true / total_set size
        total = np.count_nonzero(np.equal(classified, test_set[:, -1])) / test_set.shape[0]
        return total
