import collections
import numpy as np
from pymongo import MongoClient

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'


def make_training_and_test_sets():
  game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]
  rows = game_data.find()
  num_rows = game_data.count()
  Dataset = collections.namedtuple('Dataset', ['data', 'target'])

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
      target.append(0)
    elif row['paddle_velocity'] == 0:
      target.append(1)
    else:
      target.append(2)

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
  return Dataset(data=training_data, target=training_target), Dataset(data=test_data, target=test_target)
