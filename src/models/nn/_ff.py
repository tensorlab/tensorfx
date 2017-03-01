# Copyright 2016 TensorLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# _ff.py
# Implements FeedForwardClassification.

import enum
import math
import tensorflow as tf
import tensorfx as tfx
import tensorfx.models as models


class FeedForwardModelArguments(tfx.training.ModelArguments):
  """Arguments for feed-forward neural networks.
  """
  @classmethod
  def build_parser(cls):
    parser = super(FeedForwardModelArguments, cls).build_parser()

    optimization = parser.add_argument_group(title='Optimization',
      description='Arguments determining the optimizer behavior.')
    optimization.add_argument('--learning-rate', metavar='rate', type=float, default=0.01,
                              help='The magnitude of learning to perform at each step.')

    nn = parser.add_argument_group(title='Neural Network',
      description='Arguments controlling the structure of the neural network.')
    nn.add_argument('--hidden-layers', metavar='units', type=int, required=False,
                    action=parser.var_args_action,
                    help='The size of each hidden layer to add.')

    return parser

  def process(self):
    super(FeedForwardModelArguments, self).process()

    if self.hidden_layers:
      self.hidden_layers = map(lambda (i, s): ('layer_%d' % i, s, 'relu'),
                              enumerate(self.hidden_layers))
    else:
      self.hidden_layers = []

    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    # TODO: Add a similar customization point for evaluation - an Evaluator object


class FeedForwardClassificationArguments(FeedForwardModelArguments):
  """Arguments for feed-forward classification neural networks.
  """
  pass


class FeedForwardClassification(tfx.training.ModelBuilder):
  """A ModelBuilder for building feed-forward fully connected neural network models.

  These models are also known as multi-layer perceptrons.
  """
  def __init__(self, args, dataset):
    super(FeedForwardClassification, self).__init__(args, dataset)

    target_feature = filter(lambda f: f.type == tfx.data.FeatureType.target, dataset.features)[0]
    target_metadata = dataset.metadata[target_feature.field]

    self._classification = models.ClassificationScenario(target_metadata['entries'])

  def build_inference(self, inputs, training):
    histograms = {}
    scalars = {}

    # Build a set of hidden layers. The input to the first hidden layer is
    # the features tensor, whose shape is (batch, size).
    x = inputs['X']
    x_size = x.get_shape()[1].value

    for name, size, activation in self.args.hidden_layers:
      with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([x_size, size],
                                                  stddev=1.0 / math.sqrt(float(x_size))),
                              name='weights')
        biases = tf.Variable(tf.zeros([size]), name='biases')
        outputs = tf.nn.xw_plus_b(x, weights, biases, name='outputs')

        histograms[outputs.op.name + '.activations'] = outputs
        scalars[outputs.op.name + '.sparsity'] = tf.nn.zero_fraction(outputs)

        if activation:
          activation_fn = getattr(tf.nn, activation)
          outputs = activation_fn(outputs, name=activation)
      x = outputs
      x_size = size

    with tf.name_scope('logits'):
      weights = tf.Variable(tf.truncated_normal([x_size, self._classification.num_labels],
                                                stddev=1.0 / math.sqrt(float(x_size))),
                            name='weights')
      biases = tf.Variable(tf.zeros([self._classification.num_labels]), name='biases')
      logits = tf.nn.xw_plus_b(x, weights, biases, name='outputs')

      histograms[logits.op.name + '.activations'] = logits
      scalars[logits.op.name + '.sparsity'] = tf.nn.zero_fraction(logits)

    if training:
      with tf.name_scope(''):
        for name, t in scalars.iteritems():
          tf.summary.scalar(name, t)

        for name, t in histograms.iteritems():
          tf.summary.histogram(name, t)

        for t in tf.trainable_variables():
          tf.summary.histogram(t.op.name, t)

    return logits

  def build_training(self, global_steps, inputs, inferences):
    target_labels = inputs['Y']
    label_indices = self._classification.labels_to_indices(target_labels, one_hot=True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=inferences, labels=label_indices)

    loss = tf.reduce_mean(cross_entropy, name='loss')

    averager = tf.train.ExponentialMovingAverage(0.99)
    averaging = averager.apply([loss])

    with tf.name_scope(''):
      tf.summary.scalar('metrics/loss', loss)
      tf.summary.scalar('metrics/loss.average', averager.average(loss))

    with tf.control_dependencies([averaging]):
      gradients = self.args.optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
      train = self.args.optimizer.apply_gradients(gradients, global_steps, name='optimize')

      with tf.name_scope(''):
        for gradient, t in gradients:
          if gradient is not None:
            tf.summary.histogram(t.op.name + '.gradients', gradient)

    return loss, train

  def build_output(self, inferences):
    scores = tf.nn.softmax(inferences, name='scores')
    tf.add_to_collection('outputs', scores)

    label_indices = tf.arg_max(inferences, 1)
    labels = self._classification.indices_to_labels(label_indices)
    tf.add_to_collection('outputs', labels)

    return {
      'label': labels,
      'score': scores
    }

  def build_evaluation(self, inputs, outputs):
    target_labels = inputs['Y']
    accuracy, eval = tf.contrib.metrics.streaming_accuracy(outputs['label'], target_labels)

    with tf.name_scope(''):
      tf.summary.scalar('metrics/accuracy', accuracy)

    return accuracy, eval
