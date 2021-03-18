from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import properties

DATA_HOME = properties.DATA_HOME

import data_utils

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
  training_files_x = data_utils.get_data_files(DATA_HOME, training_records, "x")
  training_files_y = data_utils.get_data_files(DATA_HOME, training_records, "y")
  time_files_x = data_utils.get_data_files(DATA_HOME, training_records, "x_time")
  time_files_y = data_utils.get_data_files(DATA_HOME, training_records, "y_time")
  data_files = {
    "x": training_files_x,
    "y": training_files_y,
    "x_time": time_files_x,
    "y_time": time_files_y
  }
  training_stream = data_utils.DataStreamer(data_files, sample_deltas=sampling_rates["2"],
                                            do_shuffle=True, class_balancer=data_utils.BalanceSampler(),
                                            batch_size=1280)



runner()
