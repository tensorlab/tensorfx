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

# _model.py
# Implements the ModelBuilder base class.

import tensorflow as tf
from ._args import ModelArguments


def _create_interface(phase, graph, references):
  """Creates an interface instance using a dynamic type with graph and references as attributes.
  """
  interface = {'graph': graph}
  interface.update(references)

  return type(phase + 'Interface', (object,), interface)


class ModelBuilder(object):
  """Builds model graphs for different phases: training, evaluation and prediction.

  A model graph is an interface that encapsulates a TensorFlow graph, and references to tensors and
  ops within that graph.

  A ModelBuilder serves as a base class for various models. Each specific model adds its specific
  logic to build the required TensorFlow graph.
  """
  def __init__(self, args):
    """Initializes an instance of a ModelBuilder.

    Arguments:
      args: the arguments specified for training.
    """
    if args is None or not isinstance(args, ModelArguments):
      raise ValueError('args must be an instance of ModelArguments')

    self._args = args

  @property
  def args(self):
    """Retrieves the set of arguments specified for training.
    """
    return self._args

  def training(self, dataset):
    """Builds the training graph to use for training a model.

    Arguments:
      dataset: The DataSet providing a reference to training data.
    Returns:
      A training interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_training_graph(dataset)
      return _create_interface('Training', graph, references)

  def evaluation(self, dataset):
    """Builds the evaluation graph to use for evaluating a model.

    Arguments:
      dataset: The DataSet providing a reference to evaluation data.
    Returns:
      An evaluation interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_evaluation_graph(dataset)
      return _create_interface('Evaluation', graph, references)

  def prediction(self, dataset):
    """Builds the prediction graph to use for predicting with a model.

    Arguments:
      dataset: The DataSet providing schema and feature information.
    Returns:
      A prediction interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_prediction_graph(dataset)
      return _create_interface('Prediction', graph, references)

  def build_training_graph(self, dataset):
    """Builds the graph to use for training a model.

    This operates on the current default graph.

    Arguments:
      dataset: The DataSet providing a reference to training data.
    Returns:
      The set of tensors and ops references required for training.
    """
    with tf.name_scope('input'):
      # For training, ensure the data is shuffled, and don't limit to any fixed number of epochs.
      # The datasource to use is the one named as 'train' within the dataset.
      targets, features = self.build_input(dataset, 'train',
                                           batch=self.args.batch_size,
                                           epochs=self.args.epochs,
                                           shuffle=True)
    
    with tf.name_scope('inference'):
      inferences = self.build_inference(features, training=True)
    
    with tf.name_scope('train'):
      global_steps = tf.Variable(0, name='global_steps', trainable=False,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                              tf.GraphKeys.GLOBAL_STEP])
      loss, train_op = self.build_training(inferences, targets, global_steps)

    # Create the summary op that will merge all summaries across all sub-graphs
    summary_op = tf.merge_all_summaries()

    # Create the saver that will be used to save all trained variables
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.name_scope('initialization'):
      init_op = self.build_init()

    return {
      'global_steps': global_steps,
      'loss': loss,
      'init_op': init_op,
      'train_op': train_op,
      'summary_op': summary_op,
      'saver': saver
    }

  def build_evaluation_graph(self, dataset):
    """Builds the graph to use for evaluating a model during training.

    Arguments:
      dataset: The DataSet providing a reference to evaluation data.
    Returns:
      The set of tensors and ops references required for evaluation.
    """
    with tf.name_scope('input'):
      # For evaluation, compute the eval metric over a single pass over the evaluation data,
      # and avoid any overhead from shuffling.
      # The datasource to use is the one named as 'eval' within the dataset.
      targets, features = self.build_input(dataset, 'eval', batch=1, epochs=1, shuffle=False)

    with tf.name_scope('inference'):
      inferences = self.build_inference(features, training=False)

    with tf.name_scope('output'):
      outputs = self.build_output(inferences)

    with tf.name_scope('evaluation'):
      metric, eval_op = self.build_evaluation(outputs, targets)

    # Create the summary op that will merge all summaries across all sub-graphs
    summary_op = tf.merge_all_summaries()

    # Create the saver that will be used to restore all trained variables
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.name_scope('initialization'):
      init_op = self.build_init()

    return {
      'metric': metric,
      'init_op': init_op,
      'eval_op': eval_op,
      'summary_op': summary_op,
      'saver': saver
    }

  def build_prediction_graph(self, dataset):
    """Builds the graph to use for predictions with the trained model.

    Returns:
      The set of tensors and ops references required for prediction.
    """
    with tf.name_scope('input'):
      _, features = self.build_input(dataset, source=None, batch=0, epochs=0, shuffle=False)

    with tf.name_scope('inference'):
      inferences = self.build_inference(features, training=False)

    with tf.name_scope('output'):
      outputs = self.build_output(inferences)

    # Create the saver that will be used to restore all trained variables
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.name_scope('initialization'):
      init_op = self.build_init()

    return {
      'init_op': init_op,
      'saver': saver
    }

  def build_init(self):
    """Builds the initialization sub-graph

    The default implementation creates an initialization op that initializes all variables,
    locals, and tables.

    Returns:
      The init op to use to initialize the graph.
    """
    init_variables = tf.initialize_all_variables()
    init_locals = tf.initialize_local_variables()

    return tf.group(init_variables, init_locals)

  def build_input(self, dataset, source, batch, epochs, shuffle):
    """Builds the input sub-graph.

    Arguments:
      dataset: The DataSet providing containing the specified datasource.
      source: the name of data source to use for input (for training and evaluation).
      batch: the number of instances to read per batch.
      epochs: the number of passes over the data.
      shuffle: whether to shuffle the data.
    Returns:
      A tuple of targets and a dictionary of tensors key'd by feature names.
    """
    if source:
      instances = dataset[source].read(batch=batch, shuffle=shuffle, epochs=epochs)
      return dataset.parse_instances(instances)
    else:
      instances = tf.placeholder(dtype=tf.string, shape=(None,), name='instances')
      return dataset.parse_instances(instances, prediction=True)

  def build_inference(self, features, training):
    """Builds the inference sub-graph.

    Arguments:
      features: the dictionary of tensors corresponding to the input.
      training: whether the inference sub-graph is being built for the training graph.
    Returns:
      The inference values.
    """
    raise NotImplementedError('build_inference must be implemented in a derived class.')

  def build_training(self, inferences, targets, global_steps):
    """Builds the training sub-graph.

    Arguments:
      inferences: the inference values.
      targets: the target values to compare inferences to.
      global_steps: the global steps variable to use.
    Returns:
      The loss tensor, and the training op.
    """
    raise NotImplementedError('build_training must be implemented in a derived class.')

  def build_output(self, inferences):
    """Builds the output sub-graph

    Arguments:
      inferences: the inference values.
    Returns:
      A dictionary consisting of the output prediction tensors.
    """
    raise NotImplementedError('build_output must be implemented in a derived class.')

  def build_evaluation(self, outputs, targets):
    """Builds the evaluation graph.abs

    Arguments:
      predictions: the dictionary containing output tensors.
      targets: the expected target values.
    Returns:
      The eval metric tensor and the eval op.
    """
    raise NotImplementedError('build_evaluation must be implemented in a derived class.')
  