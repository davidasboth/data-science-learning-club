import numpy as np
import math
from collections import defaultdict

class Feature(object):
    def __init__(self, values, is_categorical=False):
        self._is_categorical = is_categorical
        self._values = values

        # default values for discrete/continuous prob. distributions
        self._probabilities = {}
        self._mean = 0
        self._std = 0

        # calculate probability distribution or density based on values passed in
        self._calculate_probabilities()

    def _calculate_probabilities(self):
        # in the case of a categorical variable, calculate the frequency and use them
        # to create a discrete probability distribution
        if self._is_categorical:
            # calculate value counts
            value_counts = defaultdict(int)
            for v in self._values:
                value_counts[v] += 1
            # and turn them into probabilities
            for v in np.unique(self._values):
                self._probabilities[v] = value_counts[v] / len(self._values)
        # in the case of a continuous variable, calculate mean and std for the values
        else:
            self._std = np.std(self._values)
            self._mean = np.mean(self._values)

    def get_probability(self, value):
        """
        Return the probability of this feature having a particular value
        :param value: the value to test
        :return: the probability of having that value
        """
        # for categorical variables, just return the pre-calculated probability
        # or 0 if the class is not valid
        if self._is_categorical:
            if value in self._probabilities:
                return self._probabilities[value]
            return 0.0
        else:
            # use the Gaussian function to calculate the probability
            var = float(self._std)**2
            denom = (2 * math.pi * var)**.5
            num = math.exp(-(float(value)-float(self._mean))**2/(2*var))
            return num/denom
