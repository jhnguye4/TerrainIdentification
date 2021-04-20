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
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.optimizers import Adam

import logger
import cache

LOGGER = logger.get_logger(os.path.basename(__file__.split(".")[0]))


class BaseModel:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.name = "Base Model"
    self.model = None

  def get_key(self):
    return "base_model"

  @staticmethod
  def get_class(model_name):
    return globals()[model_name]

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

  def save(self, file_path):
    parent_folder = cache.get_parent_folder(file_path)
    cache.mkdir(parent_folder)
    self.model.save(file_path)

class LogReg(BaseModel):
  def __init__(self, x, y, max_iter=1000):
    BaseModel.__init__(self, x, y)
    self.name = "Logistic Regression"
    self.model = LogisticRegression(max_iter=max_iter)

  def get_key(self):
    return "log_reg"


class KNN(BaseModel):
  def __init__(self, x, y, k=3):
    BaseModel.__init__(self, x, y)
    self.name = "%d - Nearest Neighbors" % k
    self.model = KNeighborsClassifier(n_neighbors = k)

  def get_key(self):
    return "%dnn" % self.k


class NaiveBayes(BaseModel):
  def __init__(self, x, y):
    BaseModel.__init__(self, x, y)
    self.name = "Naive Bayes"
    self.model = GaussianNB()

  def get_key(self):
    return "naive_bayes"

class SimpleLSTM(BaseModel):
  def __init__(self, train=None, valid=None,
               sample_window=None, n_features=None, n_classes=None,
               batch_size=32, epochs=10,
               optimizer=Adam(), name=None):
    if train:
      BaseModel.__init__(self, train[0], train[1])
    else:
      BaseModel.__init__(self, None, None)
    self.name = "Simple LSTM" if name is None else self.name
    self.n_classes = n_classes
    self.sample_weights = train[2] if train else None
    self.valid_x = valid[0] if valid else None
    self.valid_y = valid[1] if valid else None
    self.valid_weights = valid[2] if valid else None
    self.batch_size = batch_size
    self.epochs = epochs
    if sample_window:
      self.model = Sequential()
      self.model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(sample_window, n_features)))
      self.model.add(TimeDistributed(Dense(64, activation='relu')))
      self.model.add(Flatten())
      self.model.add(Dense(64))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(n_classes, activation='softmax'))
      self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    else:
      self.model = None

  def get_key(self):
    return "simple_lstm"

  def fit(self):
    validation_data = (self.valid_x, self.valid_y, self.valid_weights) if self.valid_weights is not None else (self.valid_x, self.valid_y)
    self.model.fit(self.x, self.y, sample_weight=self.sample_weights, batch_size=self.batch_size, epochs=self.epochs,
                   validation_data=validation_data, validation_batch_size=self.batch_size)

  def predict(self, x):
    return np.argmax(self.model.predict(x), axis=-1)
  @staticmethod
  def load(file_path, summarize=False):
    lstm = SimpleLSTM()
    lstm.model = load_model(file_path)
    if summarize:
      lstm.model.summary()
    return lstm


class BiDirectionalLSTM(BaseModel):
  def __init__(self, train=None, valid=None,
               sample_window=None, n_features=None, n_classes=None,
               batch_size=100, epochs=10,
               optimizer=Adam(), name=None):
    if train:
      BaseModel.__init__(self, train[0], train[1])
    else:
      BaseModel.__init__(self, None, None)
    self.name = "BiDirectional LSTM" if name is None else self.name
    self.n_classes = n_classes
    self.sample_weights = train[2] if train else None
    self.valid_x = valid[0] if valid else None
    self.valid_y = valid[1] if valid else None
    self.valid_weights = valid[2] if valid else None
    self.batch_size = batch_size
    self.epochs = epochs
    if sample_window:
      self.model = Sequential()
      self.model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True, input_shape=(sample_window, n_features))))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Flatten())
      self.model.add(Dense(64, activation='relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(n_classes, activation='softmax'))
      self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    else:
      self.model = None

  def get_key(self):
    return "bidirectional_lstm"

  def fit(self):
    validation_data = (self.valid_x, self.valid_y, self.valid_weights) if self.valid_weights is not None else (self.valid_x, self.valid_y)
    self.model.fit(self.x, self.y, sample_weight=self.sample_weights, batch_size=self.batch_size, epochs=self.epochs,
                   validation_data=validation_data, validation_batch_size=self.batch_size)

  def predict(self, x):
    return np.argmax(self.model.predict(x), axis=-1)

  @staticmethod
  def load(file_path, summarize=False):
    lstm = BiDirectionalLSTM()
    lstm.model = load_model(file_path)
    if summarize:
      lstm.model.summary()
    return lstm


class SuccessfulLSTM(BaseModel):
  def __init__(self, train=None, valid=None,
               sample_window=None, n_features=None, n_classes=None,
               batch_size=32, epochs=10,
               optimizer=Adam(), hidden_layer=256, name=None):

    if train:
      BaseModel.__init__(self, train[0], train[1])
    else:
      BaseModel.__init__(self, None, None)
    self.name = "BiDirectional LSTM" if name is None else self.name
    self.n_classes = n_classes
    self.sample_weights = train[2] if train else None
    self.valid_x = valid[0] if valid else None
    self.valid_y = valid[1] if valid else None
    self.valid_weights = valid[2] if valid else None
    self.batch_size = batch_size
    self.epochs = epochs
    self.hidden_layers = hidden_layer
    if sample_window:
      self.model = Sequential()
      self.model.add(
        Bidirectional(
          LSTM(hidden_layer, return_sequences=True, input_shape=(sample_window, n_features))
        )
      )
      self.model.add(Dropout(0.5))
      self.model.add(TimeDistributed(Dense(units=hidden_layer/2, activation='tanh')))
      self.model.add(Flatten())
      self.model.add(Dense(n_classes, activation='softmax'))
      self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    else:
      self.model = None

  def get_key(self):
    return "successful_lstm"

  def fit(self):
    validation_data = (self.valid_x, self.valid_y, self.valid_weights) if self.valid_weights is not None else (
    self.valid_x, self.valid_y)
    self.model.fit(self.x, self.y, sample_weight=self.sample_weights, batch_size=self.batch_size, epochs=self.epochs,
                   validation_data=validation_data, validation_batch_size=self.batch_size)

  def predict(self, x):
    return np.argmax(self.model.predict(x), axis=-1)


  @staticmethod
  def load(file_path, summarize=False):
    lstm = SuccessfulLSTM()
    lstm.model = load_model(file_path)
    if summarize:
      lstm.model.summary()
    return lstm


class VariationalLSTM(SuccessfulLSTM):
  def __init__(self, train=None, valid=None,
               sample_window=None, n_features=None, n_classes=None,
               batch_size=32, epochs=10,
               optimizer=Adam(), hidden_layer=256, name=None):
    SuccessfulLSTM.__init__(self, train=train, valid=valid, sample_window=sample_window, n_features=n_features,
                            n_classes=n_classes, batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                            hidden_layer=hidden_layer, name=name)

  def get_key(self):
    return "variational_lstm_h-%d" % self.hidden_layers

  @staticmethod
  def load(file_path, summarize=False):
    lstm = VariationalLSTM()
    lstm.model = load_model(file_path)
    if summarize:
      lstm.model.summary()
    return lstm