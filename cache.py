from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

import pickle
import joblib


def get_parent_folder(file_name):
  splits = file_name.rsplit(os.path.sep, 1)
  if len(splits) > 1:
    return splits[0]
  return None


def file_exists(file_name):
  """
  Check if file or folder exists
  :param file_name: Path of the file
  :return: True/False
  """
  return os.path.exists(file_name)


def mkdir(directory):
  """
  Create Directory if it does not exist
  """
  if not file_exists(directory):
    try:
      os.makedirs(directory)
    except OSError as e:
      if e.errno != os.errno.EEXIST:
        raise

def get_file_name(file_path):
  """
  Return name of the file from the path
  :param file_path: Path of the file
  :return: Name of the file
  """
  return file_path.rsplit(os.path.sep, 1)[-1].split(".", 1)[0]


def load_pickle(file_name, verbose=False):
  """
  :return: Content of file_name
  """
  if not file_exists(file_name):
    if verbose:
      print("File %s does not exist" % file_name)
    return None
  try:
    with open(file_name, "rb") as f:
      return pickle.load(f)
  except Exception:
    if verbose:
      print("Exception while loading file" % file_name)
    return None


def save_pickle(file_name, obj):
  """
  Save obj in file_name
  :param file_name:
  :param obj:
  :return:
  """
  parent = get_parent_folder(file_name)
  if parent:
    mkdir(parent)
  with open(file_name, "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def load_as_joblib(file_name, verbose=False):
  """
  :return: Content of file_name
  """
  if not file_exists(file_name):
    if verbose:
      print("File %s does not exist" % file_name)
    return None
  try:
    return joblib.load(file_name)
  except Exception:
    if verbose:
      print("Exception while loading file" % file_name)
    return None


def save_as_joblib(file_name, obj):
  """
  Save obj in file_name
  :param file_name:
  :param obj:
  :return:
  """
  parent = get_parent_folder(file_name)
  if parent:
    mkdir(parent)
  joblib.dump(obj, file_name)