# From https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html#load-the-iris-csv-data-to-tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from load_data import make_training_and_test_sets

import tensorflow as tf

def train_model():
  training_set, test_set = make_training_and_test_sets()

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/pong_model")

  # Fit model
  classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(x=test_set.data,
                                       y=test_set.target)["accuracy"]
  print('Accuracy: {0:f}'.format(accuracy_score))
