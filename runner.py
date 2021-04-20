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

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



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


def plot_data_splits(sampling_rate_key):
  train_valid_test_split_png = os.path.join("results", "train_valid_test_split.png")
  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 0),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom', rotation=45, fontsize=10)

  sampling_rate = sampling_rates[sampling_rate_key]
  training_data_files = data_utils.get_data_files(DATA_HOME, training_records)
  training_stream = data_utils.DataStreamer(training_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                            class_balancer=None, batch_size=1)
  train_y = training_stream.labels
  validation_data_files = data_utils.get_data_files(DATA_HOME, validation_records)
  validation_stream = data_utils.DataStreamer(validation_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                              class_balancer=None, batch_size=1)
  valid_y = validation_stream.labels
  test_data_files = data_utils.get_data_files(DATA_HOME, test_records)
  test_stream = data_utils.DataStreamer(test_data_files, sample_deltas=sampling_rate, do_shuffle=True,
                                              class_balancer=None, batch_size=1)
  test_y = test_stream.labels
  train_counter = Counter(train_y)
  valid_counter = Counter(valid_y)
  test_counter = Counter(test_y)
  labels = sorted(train_counter.keys())
  train_counts, valid_counts, test_counts = [], [], []
  for label in labels:
    train_counts.append(train_counter[label])
    valid_counts.append(valid_counter[label])
    test_counts.append(test_counter[label])
  x = np.arange(len(labels))  # the label locations
  width = 0.25  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - 4*width / 3, train_counts, width, label='Training')
  rects2 = ax.bar(x - width / 3, valid_counts, width, label='Validation')
  rects3 = ax.bar(x + 2*width / 3, test_counts, width, label='Testing')
  ax.set_ylabel('Class count')
  ax.set_yscale('log')
  ax.set_ylabel('# Classes')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_xlabel('Class labels')
  ax.set_title('Frequency of label in training, validation and testing sets')
  ax.legend()
  autolabel(rects1)
  autolabel(rects2)
  autolabel(rects3)
  fig.tight_layout()
  plt.savefig(train_valid_test_split_png)
  plt.clf()
  LOGGER.info("Chart saved in '%s'" % train_valid_test_split_png)


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


def compute_output_metrics(mdl_name, sample_rate_key, n_classes=4, do_weight=False, hidden_units=None):
  def fmt(vals):
    _fmt = "%0.4f"
    if isinstance(vals, list) or isinstance(vals, np.ndarray):
      return list(map(lambda a: _fmt % a, vals))
    return _fmt % vals
  print("\n## Model Name: %s; Sample Rate: %s" % (mdl_name, sample_rate_key) )
  mdl = model_utils.BaseModel.get_class(mdl_name)
  inst = mdl()
  key = inst.get_key()
  if hidden_units is not None:
    key = "%s_h-%d" % (key, hidden_units)
  model_path = os.path.join("models", "%s_%s.mdl" % (key, sample_rate_key))
  sampling_rate = sampling_rates[sample_rate_key]
  datasets = [("train", training_records), ("valid", validation_records), ("test", test_records)]
  model = mdl.load(model_path)
  parent_folder = os.path.join("results", cache.get_file_name(model_path))
  cache.mkdir(parent_folder)
  for dataset, data_records in datasets[::-1]:
    print("#### %s"% dataset.upper())
    dataset_files = data_utils.get_data_files(DATA_HOME, data_records, skip_y=False)
    dataset_stream = data_utils.DataStreamer(dataset_files, sample_deltas=sampling_rate, do_shuffle=False)
    x, y, sample_weights = dataset_stream.preprocess(n_classes)
    y_predicted = list(model.predict(x))
    y_actual = list(map(int, dataset_stream.labels))
    labels = list(range(n_classes))
    cm = metrics.confusion_matrix(y_actual, y_predicted, labels=labels, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(cm)
    cm_plt = disp.plot(include_values=True, cmap=plt.cm.Blues, colorbar=True)
    cm_plt_path = os.path.join(parent_folder, "cm_%s.png" % dataset)
    plt.grid(False)
    plt.savefig(cm_plt_path)
    plt.clf()
    average = "weighted" if do_weight else None
    print("%10s: %s" % ("Accuracy", fmt(metrics.accuracy_score(y_actual, y_predicted))))
    print("%10s: %s" % ("Precision", fmt(metrics.precision_score(y_actual, y_predicted, labels=labels, average=average))))
    print("%10s: %s" % ("Recall", fmt(metrics.recall_score(y_actual, y_predicted, labels=labels, average=average))))
    print("%10s: %s" % ("F1", fmt(metrics.f1_score(y_actual, y_predicted, labels=labels, average=average))))



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
  ranges = ["4", "6", "10", "30", "60"]
  for sample_rate_key in ranges:
    LOGGER.info("Processing for window size = %s" % sample_rate_key)
    model_path = os.path.join("models/bi_lstm_%s.mdl" % sample_rate_key)
    if cache.file_exists(model_path):
      model_utils.BiDirectionalLSTM.load(model_path)
      continue
    batch_size = 32
    n_epochs = 10
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


def successful_runner(sample_rate_key):
  LOGGER.info("Processing for window size = %s" % sample_rate_key)
  model_path = os.path.join("models/successful_lstm_%s.mdl" % sample_rate_key)
  if cache.file_exists(model_path):
    model_utils.SuccessfulLSTM.load(model_path)
    return
  batch_size = 100
  n_epochs = 10
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
  lstm = model_utils.SuccessfulLSTM((train_x, train_y, train_sample_weights),
                                       (valid_x, valid_y, valid_sample_weights),
                                       sampling_rate.window_size, training_stream.n_features,
                                       len(training_stream.classes), batch_size=batch_size, epochs=n_epochs)
  lstm.fit()
  lstm.model.summary()
  lstm.save(model_path)


def variational_runner(sample_rate_key, hidden_units):
  LOGGER.info("Processing for window size = %s" % sample_rate_key)
  model_path = os.path.join("models/variational_lstm_h-%d_%s.mdl" % (hidden_units, sample_rate_key))
  if cache.file_exists(model_path):
    model_utils.VariationalLSTM.load(model_path)
    return
  batch_size = 100
  n_epochs = 10
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
  lstm = model_utils.VariationalLSTM((train_x, train_y, train_sample_weights),
                                    (valid_x, valid_y, valid_sample_weights),
                                    sampling_rate.window_size, training_stream.n_features,
                                    len(training_stream.classes), batch_size=batch_size, epochs=n_epochs, hidden_layer=hidden_units)
  lstm.fit()
  lstm.model.summary()
  lstm.save(model_path)



def run_compute_output_metrics_for_sample_rates(mdl_name, sample_rates=("1", "2", "4", "6", "10", "30", "60")):
  for sample_rate in sorted(sample_rates, key=lambda x: int(x)):
    compute_output_metrics(mdl_name, sample_rate)



def model_predictor(sample_rate_key):
  sampling_rate = sampling_rates[sample_rate_key]
  model_path = os.path.join("models/successful_lstm_%s.mdl" % sample_rate_key)
  model = model_utils.BiDirectionalLSTM.load(model_path)
  for test_record in test_records:
    LOGGER.info("Predicting for '%s' ... " % test_record)
    testing_data_file = data_utils.get_data_files(TEST_HOME, [test_record], skip_y=True)
    testing_stream = data_utils.DataStreamer(testing_data_file, sample_deltas=sampling_rate, do_shuffle=False)
    test_x = testing_stream.features
    y_predicted = model.predict(test_x)
    test_file_path = os.path.join(TEST_HOME, "%sy_prediction.csv" % test_record)
    data_utils.dump_labels_to_csv(y_predicted, test_file_path)



# run_compute_output_metrics_for_sample_rates("SuccessfulLSTM", ["60"])
variational_runner("60", 32)
variational_runner("60", 64)
variational_runner("60", 128)
# successful_runner("1")
# successful_runner("2")
# successful_runner("4")
# successful_runner("6")
# successful_runner("10")
# compute_output_metrics("/Users/panzer/NCSU/Neural Networks/Competition/TerrainIdentification/models/successful_lstm_30.mdl", "30")
# bilstm_runner()
# compute_output_metrics("3nn")