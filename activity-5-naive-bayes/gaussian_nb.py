import numpy as np
import pandas as pd
from collections import defaultdict

from feature import Feature

class GaussianNaiveBayes(object):

    def __init__(self, training_set, target_values, priors=[],
                 assume_uniform=False, categoricals=[], debug_mode=False):
        # setup class variables
        if not isinstance(training_set, pd.DataFrame):
            raise TypeError("The training_set variable must be a pandas DataFrame!")
        self._X = training_set
        self._Y = target_values
        self._classes = np.unique(target_values)
        self._assume_uniform = assume_uniform
        self._debug_mode = debug_mode

        # if priors supplied, make sure the number of probabilities = number of unique classes
        if len(priors) == 0 or len(priors) == len(self._classes):
            self._priors = priors
        else:
            raise ValueError("Number of supplied priors does not match the number of classes in target_values")

        # if an array of booleans supplied for categorical columns,
        # ensure the length matches the number of features
        if len(categoricals) == 0 or len(categoricals) == len(training_set.columns):
            self._categoricals = categoricals
        else:
            raise ValueError("Number of categoricals does not match the number of features")

        # setup other variables and do necessary processing
        self._class_probabilities = {}
        self._conditional_probabilities = defaultdict(list)

    def _process_features(self):
        # for each feature, create a Feature object
        categoricals_provided = len(self._categoricals) > 0
        # get conditional probabilities by class
        for c in self._classes:
            # get the items from the training set that were in class c
            idx = [x==c for x in self._Y]
            class_data = self._X.iloc[idx]
            i = 0
            for f in self._X.columns:
                if categoricals_provided:
                    # if the categorical columns were passed in, use that knowledge
                    self._conditional_probabilities[c].append(Feature(class_data[f], is_categorical=self._categoricals[i]))
                else:
                    self._conditional_probabilities[c].append(Feature(class_data[f]))
                i += 1

    def _calculate_class_probabilities(self):
        # use priors if supplied
        if len(self._priors) > 0:
            # create a loop counter because priors list indices won't correspond to class labels
            i = 0
            for c in self._classes:
                self._class_probabilities[c] = self._priors[i]
                i += 1
        # otherwise if uniformity is assumed, make classes equiprobable
        elif self._assume_uniform:
            for c in self._classes:
                self._class_probabilities[c] = 1.0 / len(self._classes)
        # otherwise get the probabilities from the actual distribution of classes
        else:
            class_totals = defaultdict(int)
            for y in self._Y:
                class_totals[y] += 1
            for c in self._classes:
                self._class_probabilities[c] = class_totals[c] / len(self._Y)

    def train(self):
        # Step 1 - calculate class probabilities
        if(self._debug_mode):
            print("Calculating class probabilities...")

        self._calculate_class_probabilities()

        # Step 2 - calculate feature probability distributions
        if(self._debug_mode):
            print("Calculating feature probability distributions...")
        self._process_features()

    def _predict_single(self, data_point, columns):
        if self._debug_mode:
            print("Predicting data point...")
        probabilities = []
        # for each class work out the probability of P(C|X) with the naive assumptions
        for c in self._classes:
            class_prob = 1
            i = 0
            if self._debug_mode:
                print("\tCalculating P(C|X) for class {}".format(c))
            for f in columns:
                feature_prob = self._conditional_probabilities[c][i].get_probability(data_point[f])
                if self._debug_mode:
                    print("\t\tP({}={} | C={}) = {}".format(f, data_point[f], c, feature_prob))
                class_prob *= feature_prob
                i += 1
            class_prob *= self._class_probabilities[c]
            if self._debug_mode:
                print("\t\tP(C={}) = {}".format(c, self._class_probabilities[c]))
                print("\t\tP(C|X) = {}".format(class_prob))
            probabilities.append(class_prob)
        # find the maximum value and return the corresponding class
        return self._classes[np.argmax(probabilities)]


    def predict(self, data):
        """
        Make a prediction on a set of features, i.e. a DataFrame
        :param data: pandas DataFrame of features
        :return: a list of predictions on the class label
        """
        if not isinstance(data, pd.DataFrame):
            return TypeError("Passed in data must be a pandas DataFrame")
        predictions = []
        data.apply(lambda x: predictions.append(self._predict_single(x, data.columns)), axis=1)
        return predictions