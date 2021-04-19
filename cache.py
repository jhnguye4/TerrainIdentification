from __future__ import print_function, division
import sys
import os

sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


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