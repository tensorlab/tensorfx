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


def _create_interface(phase, graph, references):
  """Creates an interface instance using a dynamic type with graph and references as attributes.
  """
  interface = {'graph': graph}
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
      return _create_interface('Training', graph, references)

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
    raise NotImplementedError()

  def build_evaluation_graph(self):
    """Builds the graph to use for evaluating a model during training.

    Returns:
      The set of tensors and ops references required for evaluation.
    """
    raise NotImplementedError()

  def build_prediction_graph(self):
    """Builds the graph to use for predictions with the trained model.

    Returns:
      The set of tensors and ops references required for prediction.
    """
    raise NotImplementedError()
