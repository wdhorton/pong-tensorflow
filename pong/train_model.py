# From https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html#load-the-iris-csv-data-to-tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from load_data import make_training_and_test_sets

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

def signature_fn(examples, features, predictions):
  return {}, {
    'inputs': exporter.generic_signature({'data': examples}),
    'outputs': exporter.generic_signature({'move': predictions})
  }

def train_model():
  training_set, test_set = make_training_and_test_sets()

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir=tempfile.mkdtemp())

  # Fit model
  classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(x=test_set.data,
                                       y=test_set.target)["accuracy"]
  print('Accuracy: {0:f}'.format(accuracy_score))
  classifier.export('/tmp/pong_model', signature_fn=signature_fn)

if __name__ == '__main__':
  train_model()
