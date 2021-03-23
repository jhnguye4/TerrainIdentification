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
from sklearn.metrics import f1_score

import logger

LOGGER = logger.get_logger(os.path.basename(__file__.split(".")[0]))


class BaseModel:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.name = "Base Model"
    self.model = None

  @staticmethod
  def merge(x):
    return [list(itertools.chain.from_iterable(x_i)) for x_i in x]

  def fit(self):
    x = BaseModel.merge(self.x)
    LOGGER.info("Fitting %s model ... " % self.name)
    self.model.fit(x, self.y)

  def predict(self, x):
    x = BaseModel.merge(x)
    LOGGER.info("Predicting for %s model ... " % self.name)
    return self.model.predict(x)

  def score(self, x, y):
    x = BaseModel.merge(x)
    LOGGER.info("Computing accuracy of %s ... " % self.name)
    return self.model.score(x, y)

  def f1(self, x, y):
    x = BaseModel.merge(x)
    LOGGER.info("Computing F1-score of %s ... " % self.name)
    y_pred = self.model.predict(x)
    return f1_score(y, y_pred, average='weighted')

class LogReg(BaseModel):
  def __init__(self, x, y, max_iter=1000):
    BaseModel.__init__(self, x, y)
    self.name = "Logistic Regression"
    self.model = LogisticRegression(max_iter=max_iter)


class KNN(BaseModel):
  def __init__(self, x, y, k=3):
    BaseModel.__init__(self, x, y)
    self.name = "%d - Nearest Neighbors" % k
    self.model = KNeighborsClassifier(n_neighbors = k)


class NaiveBayes(BaseModel):
  def __init__(self, x, y):
    BaseModel.__init__(self, x, y)
    self.name = "Naive Bayes"
    self.model = GaussianNB()
