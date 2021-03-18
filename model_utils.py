from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import itertools
from sklearn.linear_model import LogisticRegression

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


