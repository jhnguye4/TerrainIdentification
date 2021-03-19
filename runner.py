from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


import properties
import data_utils
import model_utils
import logger

DATA_HOME = properties.DATA_HOME
LOGGER = logger.get_logger(os.path.basename(__file__.split(".")[0]))


training_records = [
                    "subject_001_01__", "subject_001_02__", "subject_001_03__", "subject_001_04__", "subject_001_05__", "subject_001_06__", "subject_001_07__",
                    "subject_002_01__", "subject_002_02__", "subject_002_03__", "subject_002_04__",
                    "subject_003_01__", "subject_003_02__",
                    "subject_004_01__",
                    "subject_005_01__", "subject_005_02__",
                    "subject_006_01__", "subject_006_02__",
                    "subject_007_01__", "subject_007_02__", "subject_007_03__",
                    "subject_008_01__"
                    ]

validation_records = ["subject_001_08__",
                      "subject_002_05__",
                      "subject_003_03__",
                      "subject_004_02__",
                      "subject_005_03__",
                      "subject_006_03__",
                      "subject_007_04__"]



sampling_rates = {
  "1": data_utils.SamplingRate([-0.02], 0, 0, 4),
  "2": data_utils.SamplingRate([-0.045, -0.02], 0, 1, 2),
  "4": data_utils.SamplingRate([-0.07, -0.045, -0.02, 0.005], 0, 1, 0)
}


def runner():
  for balancer in [data_utils.UnderSampler(), data_utils.BalanceSampler(), data_utils.OverSampler()]:
    for key in sorted(sampling_rates.keys()):
      sampling_rate = sampling_rates[key]
      LOGGER.info("Balancer: %s; Sampling Rate: %s" % (balancer.__class__.__name__, key))
      LOGGER.info("\n#### Training")
      training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
      training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                class_balancer=balancer, batch_size=1)
      train_x = training_stream.features
      train_y = training_stream.labels

      #Logistic Regression Model
      model = model_utils.LogReg(train_x, train_y)
      model.fit()
      training_score = model.score(train_x, train_y)
      LOGGER.info("Logistic Regression Training Score: %s" % training_score)
      LOGGER.info("\n#### Validation")
      validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      valid_x = validation_stream.features
      valid_y = validation_stream.labels
      validation_score = model.score(valid_x, valid_y)
      LOGGER.info("Logistic Regression Validation Score: %s" % validation_score)
      print()

      #KNN Model
      knnModel = model_utils.Knn(train_x, train_y)
      knnModel.fit()
      knn_training_score = knnModel.score(train_x, train_y)
      LOGGER.info("KNN Training Score: %s" % knn_training_score)
      LOGGER.info("\n#### Validation")
      knn_validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      knn_validation_stream = data_utils.DataStreamer(knn_validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      knn_valid_x = knn_validation_stream.features
      knn_valid_y = knn_validation_stream.labels
      knn_validation_score = knnModel.score(knn_valid_x, knn_valid_y)
      LOGGER.info("KNN Validation Score: %s" % knn_validation_score)
      print()

      #Naive Bayes Model
      nbModel = model_utils.NaiveBayes(train_x, train_y)
      nbModel.fit()
      nb_training_score = nbModel.score(train_x, train_y)
      LOGGER.info("Naive Bayes Training Score: %s" % nb_training_score)
      LOGGER.info("\n#### Validation")
      nb_validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      nb_validation_stream = data_utils.DataStreamer(nb_validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      nb_valid_x = nb_validation_stream.features
      nb_valid_y = nb_validation_stream.labels
      nb_validation_score = nbModel.score(nb_valid_x, nb_valid_y)
      LOGGER.info("Naive Bayes Validation Score: %s" % nb_validation_score)
      print()
runner()
