import collections
import numpy as np
from pymongo import MongoClient

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'

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

def make_training_and_test_sets(one_hot=False):
  game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]
  rows = game_data.find()
  num_rows = game_data.count()

  training_data, training_target = [], []
  test_data, test_target = [], []
  for i, row in enumerate(rows):
    if i < int(num_rows * 0.8):
      target = training_target
      data = training_data
    else:
      target = test_target
      data = test_data

    # Classes: UP -- 0, STATIONARY -- 1, DOWN -- 2
    if row['paddle_velocity'] < 0:
      target.append(0 if not one_hot else np.array([1, 0, 0]))
    elif row['paddle_velocity'] == 0:
      target.append(1 if not one_hot else np.array([0, 1, 0]))
    else:
      target.append(2 if not one_hot else np.array([0, 0, 1]))

    row_data = [
      row["ball_x_velocity"],
      row["ball_y_velocity"],
      row["ball_x_position"],
      row["ball_y_position"],
    	row["paddle_position"]
    ]
    data.append(np.asarray(row_data, dtype=np.float32))

  training_target = np.array(training_target, dtype=np.int)
  training_data = np.array(training_data)
  test_target = np.array(test_target, dtype=np.int)
  test_data = np.array(test_data)
  return DataSet(training_data, training_target), DataSet(test_data, test_target)
