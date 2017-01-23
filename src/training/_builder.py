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

# _builder.py
# Implements the GraphBuilder base class.

import tensorflow as tf


def _create_interface(phase, graph, references):
  """Creates an interface instance using a dynamic type with graph and references as attributes.
  """
  interface = {'graph': graph}
  interface.update(references)

  return type(phase + 'Interface', (object,), interface)


class GraphBuilder(object):
  """Builds model graphs for different phases: training, evaluation and prediction.

  A model graph is an interface that encapsulates a TensorFlow graph, and references to tensors and
  ops within that graph.

  A GraphBuilder serves as a base class for various models. Each specific model adds its specific
  logic to build the required TensorFlow graph.
  """
  def training(self, args, dataset):
    """Builds the training graph to use for training a model.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
      dataset: Provides the input training data.
    Returns:
      A training interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_training_graph(args, dataset)
      return _create_interface('Training', graph, references)

  def evaluation(self, args, dataset):
    """Builds the evaluation graph to use for evaluating a model.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
      dataset: Provides the input evaluation data.
    Returns:
      An evaluation interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_evaluation_graph(args, dataset)
      return _create_interface('Evaluation', graph, references)

  def prediction(self, args):
    """Builds the prediction graph to use for predicting with a model.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
    Returns:
      A prediction interface consisting of a TensorFlow graph and associated tensors and ops.
    """
    with tf.Graph().as_default() as graph:
      references = self.build_prediction_graph(args)
      return _create_interface('Prediction', graph, references)

  def build_training_graph(self, args, dataset):
    """Builds the graph to use for training a model.

    This operates on the current default graph.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
      dataset: Provides the input training data.
    Returns:
      The set of tensors and ops references required for training.
    """
    raise NotImplementedError()

  def build_evaluation_graph(self, args, dataset):
    """Builds the graph to use for evaluating a model during training.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
      dataset: Provides the input evaluation data.
    Returns:
      The set of tensors and ops references required for evaluation.
    """
    raise NotImplementedError()

  def build_prediction_graph(self, args):
    """Builds the graph to use for predictions with the trained model.

    Arguments:
      args: An args object containing all training job arguments including hyperparameters.
    Returns:
      The set of tensors and ops references required for prediction.
    """
    raise NotImplementedError()
