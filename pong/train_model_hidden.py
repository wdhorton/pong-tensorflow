# adapted from https://github.com/jasonbaldridge/try-tf/blob/master/hidden.py

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from load_data import make_training_and_test_sets

# Global variables.
NUM_LABELS = 3    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

tf.app.flags.DEFINE_integer('num_epochs', 1,
                            'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_hidden', 1,
                            'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
FLAGS = tf.app.flags.FLAGS

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    training_set, test_set = make_training_and_test_sets(one_hot=True)
    # Extract it into numpy arrays.
    train_data, train_labels = training_set.data, training_set.target
    test_data, test_labels = test_set.data, test_set.target

    # Get the shape of the training data.
    train_size, num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the size of layer one.
    num_hidden = FLAGS.num_hidden

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # Initialize the hidden weights and biases.
    w_hidden = init_weights(
        [num_features, num_hidden],
        'xavier',
        xavier_params=(num_features, num_hidden))

    b_hidden = init_weights([1,num_hidden],'zeros')

    # The hidden layer.
    hidden = tf.nn.tanh(tf.matmul(x,w_hidden) + b_hidden)

    # Initialize the output weights and biases.
    w_out = init_weights(
        [num_hidden, NUM_LABELS],
        'xavier',
        xavier_params=(num_hidden, NUM_LABELS))

    b_out = init_weights([1,NUM_LABELS],'zeros')

    # The output layer.
    y = tf.nn.softmax(tf.matmul(hidden, w_out) + b_out)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
    	tf.initialize_all_variables().run()
    	if verbose:
    	    print 'Initialized!'
    	    print
    	    print 'Training.'

    	# Iterate and train.
    	for step in xrange(num_epochs * train_size // BATCH_SIZE):
    	    if verbose:
    	        print step,

    	    offset = (step * BATCH_SIZE) % train_size
    	    batch_data = train_data[offset:(offset + BATCH_SIZE), :]
    	    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    	    train_step.run(feed_dict={x: batch_data, y_: batch_labels})
    	    if verbose and offset >= train_size-BATCH_SIZE:
    	        print
    	print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

if __name__ == '__main__':
    tf.app.run()
