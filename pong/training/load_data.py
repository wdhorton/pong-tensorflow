import collections
import numpy as np
from pymongo import MongoClient
from math import sqrt

from itertools import chain
from random import random

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'
BINARY_COLLECTION_NAME = 'game_data_binary'

MAX_Y_POSITION = 400
MIN_Y_POSITION = 0
MAX_X_POSITION = 600
MIN_X_POSITION = 0
MAX_X_VELOCITY = 5
MIN_X_VELOCITY = -5
MAX_Y_VELOCITY = sqrt(50 - 9)
MIN_Y_VELOCITY = -1 * sqrt(50 - 9)

def min_max_scale(x, x_min, x_max):
  return (x - x_min) / (x_max - x_min)

def scale_features(row):
  base = [
    min_max_scale(row["ball_x_velocity"], MIN_X_VELOCITY, MAX_X_VELOCITY),
    min_max_scale(row["ball_y_velocity"], MIN_Y_VELOCITY, MAX_Y_VELOCITY),
    min_max_scale(row["ball_x_position"], MIN_X_POSITION, MAX_X_POSITION),
    min_max_scale(row["ball_y_position"], MIN_Y_POSITION, MAX_Y_POSITION),
    min_max_scale(row["paddle_position"], MIN_Y_POSITION, MAX_Y_POSITION)
  ]
  base.extend([
    base[0] * base[1],
    base[2] * base[3],
    base[0] * base[1] * base[2] * base[3],
    base[0] * base[1] * base[2] * base[3] * base[4]
  ])
  return base


# Dataset class adapted from https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/contrib/learn/python/learn/datasets/mnist.py
class DataSet(object):
  def __init__(self, data, target):
    self._num_examples = data.shape[0]
    self._data = data
    self._target = target
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def target(self):
    return self._target

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._target = self._target[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._target[start:end]

def make_training_and_test_sets(one_hot=False, balanced=False, binary=False):
  game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME if not binary else BINARY_COLLECTION_NAME]
  if balanced:
    up_rows = game_data.find({ 'paddle_velocity': { '$lt': 0 } })
    down_rows = game_data.find({ 'paddle_velocity': { '$gt': 0 } })
    stationary_rows = game_data.find({ 'paddle_velocity': 0 }).limit(max(up_rows.count(), down_rows.count()))

    rows = chain(up_rows, down_rows, stationary_rows)
    num_rows = up_rows.count() + down_rows.count() + stationary_rows.count()
  else:
    rows = game_data.find()
    num_rows = rows.count()

  training_data, training_target = [], []
  test_data, test_target = [], []
  for i, row in enumerate(rows):
    if random() < 0.8:
      target = training_target
      data = training_data
    else:
      target = test_target
      data = test_data

    # Classes: UP -- 0, STATIONARY -- 1, DOWN -- 2
    if row['paddle_velocity'] < 0:
      target.append(0 if not one_hot else (np.array([1, 0, 0]) if not binary else np.array([1, 0])))
    elif row['paddle_velocity'] == 0:
      if binary:
        raise
      target.append(1 if not one_hot else np.array([0, 1, 0]))
    else:
      target.append(2 if not one_hot else (np.array([0, 0, 1]) if not binary else np.array([0, 1])))

    row_data = scale_features(row)
    data.append(np.asarray(row_data, dtype=np.float32))

  training_target = np.array(training_target, dtype=np.int)
  training_data = np.array(training_data)
  test_target = np.array(test_target, dtype=np.int)
  test_data = np.array(test_data)
  return DataSet(training_data, training_target), DataSet(test_data, test_target)
