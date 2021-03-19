from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import logger

LOGGER = logger.get_logger(os.path.basename(__file__.split(".")[0]))


class LogReg:
  def __init__(self, x, y, max_iter=1000):
    self.x = x
    self.y = y
    self.model = LogisticRegression(max_iter=max_iter)

  @staticmethod
  def merge(x):
    return [list(itertools.chain.from_iterable(x_i)) for x_i in x]

  def fit(self):
    x = LogReg.merge(self.x)
    LOGGER.info("Fitting Logistic Regression model")
    self.model.fit(x, self.y)

  def predict(self, x):
    x = LogReg.merge(x)
    return self.model.predict(x)

  def score(self, x, y):
    x = LogReg.merge(x)
    LOGGER.info("Scoring Logistic Regression model")
    return self.model.score(x, y)

class Knn:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.model = KNeighborsClassifier(n_neighbors = 3)

  @staticmethod
  def merge(x):
    return [list(itertools.chain.from_iterable(x_i)) for x_i in x]

  def fit(self):
    x = Knn.merge(self.x)
    LOGGER.info("Fitting KNN Model")
    self.model.fit(x, self.y)

  def predict(self, x):
    x = Knn.merge(x)
    return self.model.predict(x)

  def score(self, x, y):
    x = Knn.merge(x)
    LOGGER.info("Scoring KNN Model")
    return self.model.score(x, y)

class NaiveBayes:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.model = GaussianNB()

  @staticmethod
  def merge(x):
    return [list(itertools.chain.from_iterable(x_i)) for x_i in x]

  def fit(self):
    x = NaiveBayes.merge(self.x)
    LOGGER.info("Fitting Naive Bayes Model")
    self.model.fit(x, self.y)

  def predict(self, x):
    x = NaiveBayes.merge(x)
    return self.model.predict(x)

  def score(self, x, y):
    x = NaiveBayes.merge(x)
    LOGGER.info("Scoring Naive Bayes Model")
    return self.model.score(x, y)