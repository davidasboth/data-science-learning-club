from __future__ import division
import numpy as np
import pandas as pd

class BinaryMetrics(object):
    def __init__(self, tp, fp, tn, fn):
        """
        Calculate accuracy, precision, recall and F-score
        :param tp: number of true positives
        :param fp: number of false positives
        :param tn: number of true negatives
        :param fn: number of false negatives
        :return: None
        """
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

class MultiClassMetrics(object):
    def __init__(self, correct, incorrect):
        self.accuracy = correct / (correct + incorrect)

class Evaluator(object):
    def evaluate(self, predicted, actual, binary=True):
        """
        Evaluate predictions based on actual values
        :param predicted: List or pandas Series of predicted classes
        :param actual: List or pandas Series of true classes
        :param binary: True in the case of binary classification
        :return: BinaryMetrics or MultiClassMetrics class with accuracy measures
        """
        # convert lists to pandas Series objects
        if isinstance(predicted, list):
            predicted = pd.Series(data=predicted)
        if(isinstance(actual, list)):
            actual = pd.Series(data=actual)

        if binary:
            # get true/false positives/negatives
            tp = np.where((predicted.values == 1) & (actual.values == 1), 1, 0).sum()
            fp = np.where((predicted.values == 1) & (actual.values == 0), 1, 0).sum()
            tn = np.where((predicted.values == 0) & (actual.values == 0), 1, 0).sum()
            fn = np.where((predicted.values == 0) & (actual.values == 1), 1, 0).sum()
            return BinaryMetrics(tp, fp, tn, fn)
        else:
            # for multi-class just calculate correct vs. misclassified
            correct = np.where(predicted.values == actual.values).sum()
            incorrect = np.where(predicted.values != actual.values).sum()
            return MultiClassMetrics(correct, incorrect)