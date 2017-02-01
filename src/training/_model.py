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


def _create_interface(phase, graph, references, scaffold=None, hooks=None):
  """Creates an interface instance using a dynamic type with graph and references as attributes.
  """
  interface = {'graph': graph, 'scaffold': scaffold, 'hooks': hooks}
  interface.update(references)

  return type(phase + 'Interface', (object,), interface)


class ModelBuilderArguments(object):
  """An object that defines various arguments used to build and train models.
  """
  pass


class ModelBuilder(object):
  """Builds model graphs for different phases: training, evaluation and prediction.

  A model graph is an interface that encapsulates a TensorFlow graph, and references to tensors and
  ops within that graph.

  A ModelBuilder serves as a base class for various models. Each specific model adds its specific
  logic to build the required TensorFlow graph.

  Every ModelBuilder implementation should have an associated arguments class that exists within
  the same module, and is named the same name as the ModelBuilder with an "Arguments" suffix.
  """
  def __init__(self, args, dataset, output, config):
    """Initializes an instance of a ModelBuilder.

    Arguments:
      args: the arguments specified for training.
      dataset: the input data to use during training.
      output: the output location to use during training.
      config: the training configuration.
    """
    self._args = args
    self._dataset = dataset
    self._output = output
    self._config = config

  @classmethod
  def create(cls, args, dataset, output, config):
    """Creates an instance of a ModelBuilder.

    Arguments:
      cls: the class of ModelBuilder to create.
      args: the arguments specified for training.
      dataset: the input data to use during training.
      output: the output location to use during training.
      config: the training configuration.
    """
    return cls(args, dataset, output, config)

  @property
  def args(self):
    """Retrieves the set of arguments specified for training.
    """
    return self._args

  @property
  def config(self):
    """Retrieves the training configuration.
    """
    return self._config

  @property
  def dataset(self):
    """Retrieves the input used during training.
    """
    return self._dataset

  @property
  def output(self):
    """Retrieves the output location to use during training.
    """
    return self._output

  def training(self):
    """Builds the training graph to use for training a model.

    Returns:
      A training interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_training_graph()

      # TODO: Initialize the scaffold object and hooks collection
      scaffold = tf.train.Scaffold()
      hooks = []

      return _create_interface('Training', graph, references, scaffold, hooks)

  def evaluation(self):
    """Builds the evaluation graph to use for evaluating a model.

    Returns:
      An evaluation interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_evaluation_graph()
      return _create_interface('Evaluation', graph, references)

  def prediction(self):
    """Builds the prediction graph to use for predicting with a model.

    Returns:
      A prediction interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_prediction_graph()
      return _create_interface('Prediction', graph, references)

  def build_training_graph(self):
    """Builds the graph to use for training a model.

    This operates on the current default graph.

    Returns:
      The set of tensors and ops references required for training.
    """
    with tf.name_scope('input'):
      # For training, ensure the data is shuffled.
      # The datasource to use is the one named as 'eval' within the dataset.
      features, targets = self.build_input(self._dataset.train, shuffle=True)
    
    with tf.name_scope('inference'):
      inferences = self.build_inference(features)
    
    with tf.name_scope('train'):
      global_steps = tf.Variable(0, name='global_steps', trainable=False,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                              tf.GraphKeys.GLOBAL_STEP])
      loss, train_op = self.build_training(inferences, targets, global_steps)

    # Create the summary op that will merge all summaries across all sub-graphs
    summary_op = tf.merge_all_summaries()

    # Create the saver that will be used to save all trained variables
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.name_scope('initializations'):
      init_op = self.build_init()

    return {
      'global_steps': global_steps,
      'loss': loss,
      'init_op': init_op,
      'train_op': train_op,
      'summary_op': summary_op,
      'saver': saver
    }

  def build_evaluation_graph(self):
    """Builds the graph to use for evaluating a model during training.

    Returns:
      The set of tensors and ops references required for evaluation.
    """
    with tf.name_scope('input'):
      # For evaluation, compute the eval metric over a single pass over the evaluation data,
      # and avoid any overhead from shuffling.
      # The datasource to use is the one named as 'eval' within the dataset.
      features, targets = self.build_input(self._dataset.eval, shuffle=False, epochs=1)

    with tf.name_scope('inference'):
      inferences = self.build_inference(features, training=False)

    with tf.name_scope('output'):
      predictions = self.build_output(inferences)

    with tf.name_scope('evaluation'):
      metric, eval_op = self.build_evaluation(predictions, targets)

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

  def build_prediction_graph(self):
    """Builds the graph to use for predictions with the trained model.

    Returns:
      The set of tensors and ops references required for prediction.
    """
    with tf.name_scope('input'):
      features = self.build_input()

    with tf.name_scope('inference'):
      inferences = self.build_inference(features, training=False)

    with tf.name_scope('output'):
      predictions = self.build_output(inferences)

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
    init_tables = tf.initialize_all_tables()

    return tf.group(init_variables, init_locals, init_tables)

  def build_input(self, datasource=None, shuffle=False, epochs=None):
    """Builds the input sub-graph.

    Arguments:
      datasource: the data source to use for input (for training and evaluation).
      shuffle: whether to shuffle the data.
      epochs: the number of passes over the data.
      prediction: whether the input sub-graph is being built for the prediction graph.
    Returns:
      A tuple consisting of features dictionary, and the targets (in case of training/evaluation)
    """
    # TODO: Delegate to datasource for input
    raise NotImplementedError('Implement this')

  def build_inference(self, features, training=True):
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

  def build_evaluation(self, predictions, targets):
    """Builds the evaluation graph.abs

    Arguments:
      predictions: the dictionary containing output tensors.
      targets: the expected target values.
    Returns:
      The eval metric tensor and the eval op.
    """
    raise NotImplementedError('build_evaluation must be implemented in a derived class.')
  