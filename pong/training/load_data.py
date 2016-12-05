import collections
import numpy as np
from pymongo import MongoClient
from math import sqrt

from itertools import chain
from random import random

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'
BINARY_COLLECTION_NAME = 'game_data_binary'

def min_max_scale(x, x_min, x_max):
  if x_max - x_min:
    return (x - x_min) / (x_max - x_min)

FEATURES = [
  "ball_x_velocity",
  "ball_y_velocity",
  "ball_x_position",
  "ball_y_position",
  "paddle_position",
]

MAXES = {
  "ball_x_velocity": 5,
  "ball_y_velocity": sqrt(50 - 9),
  "ball_x_position": 600,
  "ball_y_position": 400,
  "paddle_position": 400,
}

MINS = {
  "ball_x_velocity": -5,
  "ball_y_velocity": -1 * sqrt(50 - 9),
  "ball_x_position": 0,
  "ball_y_position": 0,
  "paddle_position": 0,
}

def scale_features(row):
  linear_terms = [min_max_scale(row[feature], MINS[feature], MAXES[feature]) for feature in FEATURES]
  quadratic_terms = [min_max_scale(row[feature1] * row[feature2], MINS[feature1] * MINS[feature2], MAXES[feature1] * MAXES[feature2]) for feature1 in FEATURES for feature2 in FEATURES]
  cubic_terms = [min_max_scale(row[feature1] * row[feature2] * row[feature3], MINS[feature1] * MINS[feature2] * MINS[feature3], MAXES[feature1] * MAXES[feature2] * MAXES[feature3]) for feature1 in FEATURES for feature2 in FEATURES for feature3 in FEATURES]
  return filter(lambda x: x is not None, linear_terms + quadratic_terms + cubic_terms)


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
