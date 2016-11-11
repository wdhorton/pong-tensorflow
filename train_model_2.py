# From https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist_softmax.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

from load_data import make_training_and_test_sets

def main(_):
  training_set, test_set = make_training_and_test_sets(one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 5])
  W = tf.Variable(tf.zeros([5, 3]))
  b = tf.Variable(tf.zeros([3]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 3])

  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = training_set.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: test_set.data,
                                      y_: test_set.target}))

  export_path = sys.argv[-1]
  print 'Exporting trained model to', export_path
  saver = tf.train.Saver(sharded=True)
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
      sess.graph.as_graph_def(),
      named_graph_signatures={
          'inputs': exporter.generic_signature({'data': x}),
          'outputs': exporter.generic_signature({'move': y})})
  model_exporter.export(export_path, tf.constant(1), sess)

if __name__ == '__main__':
  tf.app.run()
