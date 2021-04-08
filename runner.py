from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="darkgrid")
from collections import Counter
import numpy as np
from sklearn import metrics

import properties
import data_utils
import model_utils
import logger
import cache

DATA_HOME = properties.DATA_HOME
TEST_HOME = properties.TEST_HOME
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
                      #"subject_002_05__",
                      "subject_003_03__",
                      #"subject_004_02__",
                      "subject_005_03__",
                      #"subject_006_03__",
                      "subject_007_04__"]


test_records = [#"subject_001_08__",
                "subject_002_05__",
                #"subject_003_03__",
                "subject_004_02__",
                #"subject_005_03__",
                "subject_006_03__",
                #"subject_007_04__"
                ]


# test_records = ["subject_009_01__",
#                 "subject_010_01__",
#                 "subject_011_01__",
#                 "subject_012_01__"]


sampling_rates = {
  "1": data_utils.SamplingRate([-0.02], 0, 0, 4),
  "2": data_utils.SamplingRate([-0.045, -0.02], 0, 1, 4),
  "4": data_utils.SamplingRate([-0.07, -0.045, -0.02, 0.005], -2, 1, 4),
  "6": data_utils.SamplingRate([-0.12, -0.095, -0.07, -0.045, -0.02, 0.005], -4, 1, 4),
  "10": data_utils.SamplingRate([-0.22, -0.195, -0.17, -0.145, -0.12, -0.095, -0.07, -0.045, -0.02, 0.005], -8, 1, 4),
  "30": data_utils.SamplingRate([-0.72, -0.695, -0.67, -0.645, -0.62, -0.595, -0.57, -0.545, -0.52, -0.495, -0.47, -0.445, -0.42, -0.395, -0.37, -0.345, -0.32, -0.295, -0.27, -0.245, -0.22, -0.195, -0.17, -0.145, -0.12, -0.095, -0.07, -0.045, -0.02, 0.005], -28, 1, 4),
  "60": data_utils.SamplingRate([-1.47, -1.445, -1.42, -1.395, -1.37, -1.345, -1.32, -1.295, -1.27, -1.245, -1.22, -1.195, -1.17, -1.145, -1.12, -1.095, -1.07, -1.045, -1.02, -0.995, -0.97, -0.945, -0.92, -0.895, -0.87, -0.845, -0.82, -0.795, -0.77, -0.745, -0.72, -0.695, -0.67, -0.645, -0.62, -0.595, -0.57, -0.545, -0.52, -0.495, -0.47, -0.445, -0.42, -0.395, -0.37, -0.345, -0.32, -0.295, -0.27, -0.245, -0.22, -0.195, -0.17, -0.145, -0.12, -0.095, -0.07, -0.045, -0.02, 0.005], -58, 1, 4)
}


def logReg_runner():
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
      training_score = model.f1(train_x, train_y)
      LOGGER.info("Logistic Regression Training Score: %s" % training_score)
      LOGGER.info("\n#### Validation")
      validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      valid_x = validation_stream.features
      valid_y = validation_stream.labels
      validation_score = model.f1(valid_x, valid_y)
      LOGGER.info("Logistic Regression Validation Score: %s" % validation_score)
      print()


def knn_runner():
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

      #KNN Model
      knnModel = model_utils.KNN(train_x, train_y)
      knnModel.fit()
      knn_training_score = knnModel.f1(train_x, train_y)
      LOGGER.info("KNN Training Score: %s" % knn_training_score)
      LOGGER.info("\n#### Validation")
      knn_validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      knn_validation_stream = data_utils.DataStreamer(knn_validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      knn_valid_x = knn_validation_stream.features
      knn_valid_y = knn_validation_stream.labels
      knn_validation_score = knnModel.f1(knn_valid_x, knn_valid_y)
      LOGGER.info("KNN Validation Score: %s" % knn_validation_score)
      print()


def nb_runner():
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

      #Naive Bayes Model
      nbModel = model_utils.NaiveBayes(train_x, train_y)
      nbModel.fit()
      nb_training_score = nbModel.f1(train_x, train_y)
      LOGGER.info("Naive Bayes Training Score: %s" % nb_training_score)
      LOGGER.info("\n#### Validation")
      nb_validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
      nb_validation_stream = data_utils.DataStreamer(nb_validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                                  class_balancer=None, batch_size=1)
      nb_valid_x = nb_validation_stream.features
      nb_valid_y = nb_validation_stream.labels
      nb_validation_score = nbModel.f1(nb_valid_x, nb_valid_y)
      LOGGER.info("Naive Bayes Validation Score: %s" % nb_validation_score)
      print()


def test_runner():
  sampling_rate = sampling_rates["4"]
  balancer = data_utils.UnderSampler()
  training_data_files = data_utils.get_data_files(DATA_HOME, training_records + validation_records)
  training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                            class_balancer=balancer, batch_size=1)
  train_x = training_stream.features
  train_y = training_stream.labels
  model = model_utils.KNN(train_x, train_y)
  model.fit()
  for test_record in test_records:
    LOGGER.info("Predicting for '%s' ... " % test_record)
    testing_data_file = data_utils.get_data_files(TEST_HOME, [test_record], skip_y=True)
    testing_stream = data_utils.DataStreamer(testing_data_file, sample_deltas=sampling_rate, do_shuffle=False)
    test_x = testing_stream.features
    y_predicted = model.predict(test_x)
    test_file_path = os.path.join(TEST_HOME, "%sy_prediction.csv" % test_record)
    data_utils.dump_labels_to_csv(y_predicted, test_file_path)



def test_optimal_k_for_knn():
  sampling_rate = sampling_rates["4"]
  balancer = data_utils.UnderSampler()
  training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
  training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                            class_balancer=balancer, batch_size=1)
  train_x = training_stream.features
  train_y = training_stream.labels
  validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
  validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                              class_balancer=balancer, batch_size=1)
  valid_x = validation_stream.features
  valid_y = validation_stream.labels
  knn_scores = [["k", "f1"]]
  for k in range(1, 31):
    model = model_utils.KNN(train_x, train_y, k=k)
    model.fit()
    validation_score = model.f1(valid_x, valid_y)
    knn_scores.append([k, validation_score])
  knn_scores_file = os.path.join("results", "knn_tuned.csv")
  with open(knn_scores_file, "w") as f:
    write = csv.writer(f)
    write.writerows(knn_scores)


def plot_optimal_k():
  knn_scores_file = os.path.join("results", "knn_tuned.csv")
  knn_scores_img = os.path.join("results", "knn_tuned.png")
  data = pd.read_csv(knn_scores_file)
  fig = sns.lineplot(data=data, x="k", y="f1")
  fig.set(xlabel = "k (# nearest neighbors)", ylabel="F1 scores")
  plt.savefig(knn_scores_img)
  plt.clf()


def plot_data_splits():
  train_valid_split_png = os.path.join("results", "train_valid_split.png")
  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

  sampling_rate = sampling_rates["4"]
  training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
  training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                            class_balancer=None, batch_size=1)
  train_y = training_stream.labels
  validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
  validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                              class_balancer=None, batch_size=1)
  valid_y = validation_stream.labels
  train_counter = Counter(train_y)
  valid_counter = Counter(valid_y)
  labels = sorted(train_counter.keys())
  train_counts, valid_counts = [], []
  for label in labels:
    train_counts.append(train_counter[label])
    valid_counts.append(valid_counter[label])
  x = np.arange(len(labels))  # the label locations
  width = 0.35  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, train_counts, width, label='Training')
  rects2 = ax.bar(x + width / 2, valid_counts, width, label='Validation')
  ax.set_ylabel('Label')
  ax.set_yscale('log')
  ax.set_title('Frequency of label in training and validation sets')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()
  autolabel(rects1)
  autolabel(rects2)
  fig.tight_layout()
  plt.savefig(train_valid_split_png)
  plt.clf()
  LOGGER.info("Chart saved in '%s'" % train_valid_split_png)


def output_predicted():
  sampling_rate = sampling_rates["4"]
  balancer = data_utils.OverSampler()
  training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
  training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                            class_balancer=balancer, batch_size=1)
  train_x = training_stream.features
  train_y = training_stream.labels
  model = model_utils.KNN(train_x, train_y, k=3)
  model.fit()
  validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
  validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                              class_balancer=None, batch_size=1)
  valid_x = validation_stream.features
  valid_y = validation_stream.labels
  y_predicted = model.predict(valid_x)
  test_file_path = os.path.join("results", "%s_prediction.csv" % "3nn")
  data_utils.dump_labels_to_csv(y_predicted, test_file_path)


def compute_output_metrics(model_name):
  sampling_rate = sampling_rates["4"]
  test_file_path = os.path.join("results", "%s_prediction.csv" % model_name)
  y_predicted = list(map(int, data_utils.read_csv_file(test_file_path, as_singles=True)))
  validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
  validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                              class_balancer=None, batch_size=1)
  valid_y = list(map(int, validation_stream.labels))
  print(y_predicted[:10])
  print(valid_y[:10])
  labels = [0,1,2,3]
  print(metrics.precision_recall_fscore_support(valid_y, y_predicted, labels=labels))
  print(metrics.precision_recall_fscore_support(valid_y, y_predicted, labels=labels, average="weighted"))
  cm = metrics.confusion_matrix(valid_y, y_predicted, labels=labels, normalize='true')
  disp = metrics.ConfusionMatrixDisplay(cm)

  cm_plt = disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True)
  cm_plt_path = os.path.join("results", "%s_cm.png" % model_name)
  plt.grid(False)
  plt.savefig(cm_plt_path)
  print("Accuracy Score: %f" % metrics.accuracy_score(valid_y, y_predicted))


def preview_data_features():
  sampling_rate = sampling_rates["4"]
  data_files = data_utils.get_data_files(DATA_HOME, training_records + validation_records)
  stream = data_utils.DataStreamer(data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                            class_balancer=None, batch_size=1)
  features = stream.features
  data_list = []
  for feature_set, label in zip(features, stream.labels):
    label = str(int(label))
    for feature in feature_set:
      f_sum = sum(feature)
      feature = list(feature)
      if f_sum <= data_utils.FLOAT_ERROR:
        continue
      data_list.append(feature + [label])
  print(len(data_list))
  feature_names = [0, 1, 2, 3, 4, 5]
  classes = ["0", "1", "2", "3"]
  df_names = feature_names + ["class"]
  df = pd.DataFrame(data_list, columns=df_names)
  grouped = df.groupby('class')
  for clazz in classes:
    print("Class: %s " % clazz)
    df = grouped.get_group(clazz)
    print(df.mean())
    print(df.std())
    print("*" * 20)

  # plt.savefig(os.path.join("results","data_hist.png"))


def lstm_runner():
  for sample_rate_key in ["4", "6", "10", "30", "60"]:
    LOGGER.info("Processing for window size = %s" % sample_rate_key)
    model_path = os.path.join("models/simple_lstm_%s.mdl" % sample_rate_key)
    if cache.file_exists(model_path):
      model_utils.SimpleLSTM.load(model_path)
      continue
    batch_size = 32
    n_epochs = 20
    sampling_rate = sampling_rates[sample_rate_key]
    # balancer = data_utils.OverSampler()
    balancer = None
    training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
    training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                              class_balancer=balancer, batch_size=1)
    train_x, train_y, train_sample_weights = training_stream.preprocess()
    validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
    validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                                class_balancer=None, batch_size=1)
    valid_x, valid_y, valid_sample_weights = validation_stream.preprocess(n_classes=len(training_stream.classes))
    lstm = model_utils.SimpleLSTM((train_x, train_y, train_sample_weights), (valid_x, valid_y, valid_sample_weights),
                                  sampling_rate.window_size, training_stream.n_features,
                                  len(training_stream.classes), batch_size=batch_size, epochs=n_epochs)
    lstm.model.summary()
    lstm.fit()
    lstm.save(model_path)



def bilstm_runner():
  for sample_rate_key in ["4", "6", "10", "30", "60"]:
    LOGGER.info("Processing for window size = %s" % sample_rate_key)
    model_path = os.path.join("models/bi_lstm_%s.mdl" % sample_rate_key)
    if cache.file_exists(model_path):
      model_utils.BiDirectionalLSTM.load(model_path)
      continue
    batch_size = 32
    n_epochs = 20
    sampling_rate = sampling_rates[sample_rate_key]
    # balancer = data_utils.OverSampler()
    balancer = None
    training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
    training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                              class_balancer=balancer, batch_size=1)
    train_x, train_y, train_sample_weights = training_stream.preprocess()
    validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
    validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                                class_balancer=None, batch_size=1)
    valid_x, valid_y, valid_sample_weights = validation_stream.preprocess(n_classes=len(training_stream.classes))
    lstm = model_utils.BiDirectionalLSTM((train_x, train_y, train_sample_weights), (valid_x, valid_y, valid_sample_weights),
                                  sampling_rate.window_size, training_stream.n_features,
                                  len(training_stream.classes), batch_size=batch_size, epochs=n_epochs)
    lstm.fit()
    lstm.model.summary()
    lstm.save(model_path)


def test():
  sampling_rate = sampling_rates["4"]
  data_files = data_utils.get_data_files(DATA_HOME, training_records + validation_records)
  stream = data_utils.DataStreamer(data_files, sample_deltas=sampling_rate, do_shuffle=False,
                                   class_balancer=None, batch_size=1)
  x, y = stream.preprocess()
  print(x)
  print(y)


bilstm_runner()
# compute_output_metrics("3nn")