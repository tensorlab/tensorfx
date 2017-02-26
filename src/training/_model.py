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
import tensorfx as tfx
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
  def __init__(self, args, dataset):
    """Initializes an instance of a ModelBuilder.

    Arguments:
      args: the arguments specified for training.
      dataset: the DataSet providing training and evaluation data.
    """
    if args is None or not isinstance(args, ModelArguments):
      raise ValueError('args must be an instance of ModelArguments')

    self._args = args
    self._dataset = dataset

  @property
  def args(self):
    """Retrieves the set of arguments specified for training.
    """
    return self._args

  @property
  def dataset(self):
    """Retrieves the DataSet being used for training and evaluation data.
    """
    return self._dataset

  def build_graph_interfaces(self, config):
    """Builds graph interfaces for training and evaluating a model, and for predicting using it.

    A graph interface is an object containing a TensorFlow graph member, as well as members
    corresponding to various tensors and ops within the graph.

    Arguments:
      config: The training Configuration object.
    Returns:
      A tuple consisting of the training, evaluation and prediction interfaces.
    """
    with tf.Graph().as_default() as graph:
      with tf.device(config.create_device_setter(self._args)):
        references = self.build_training_graph()
        training = _create_interface('Training', graph, references)

    with tf.Graph().as_default() as graph:
      references = self.build_evaluation_graph()
      evaluation = _create_interface('Evaluation', graph, references)

    with tf.Graph().as_default() as graph:
      references = self.build_prediction_graph()
      prediction = _create_interface('Prediction', graph, references)

    return training, evaluation, prediction

  def build_training_graph(self):
    """Builds the graph to use for training a model.

    This operates on the current default graph.

    Returns:
      The set of tensors and ops references required for training.
    """
    with tf.name_scope('input'):
      # For training, ensure the data is shuffled, and don't limit to any fixed number of epochs.
      # The datasource to use is the one named as 'train' within the dataset.
      inputs = self.build_input('train',
                                batch=self.args.batch_size,
                                epochs=self.args.epochs,
                                shuffle=True)
    
    with tf.name_scope('inference'):
      inferences = self.build_inference(inputs, training=True)

    with tf.name_scope('train'):
      # Global steps is marked as trainable (explicitly), so as to have it be saved into checkpoints
      # for the purposes of resumed training.
      global_steps = tf.Variable(0, name='global_steps', dtype=tf.int64, trainable=True,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                              tf.GraphKeys.GLOBAL_STEP,
                                              tf.GraphKeys.TRAINABLE_VARIABLES])
      loss, train_op = self.build_training(global_steps, inputs, inferences)

    with tf.name_scope('initialization'):
      # Create the saver that will be used to save and restore (in cases of resumed training)
      # trained variables.
      saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

      init_op, local_init_op = self.build_init()
      ready_op = tf.report_uninitialized_variables(tf.trainable_variables())

    # Create the summary op that will merge all summaries across all sub-graphs
    summary_op = tf.summary.merge_all()

    scaffold = tf.train.Scaffold(init_op=init_op,
                                 local_init_op=local_init_op,
                                 ready_op=ready_op,
                                 ready_for_local_init_op=ready_op,
                                 summary_op=summary_op,
                                 saver=saver)
    scaffold.finalize()

    return {
      'global_steps': global_steps,
      'loss': loss,
      'init_op': init_op,
      'local_init_op': local_init_op,
      'ready_op': ready_op,
      'train_op': train_op,
      'summary_op': summary_op,
      'saver': saver,
      'scaffold': scaffold
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
      inputs = self.build_input('eval', batch=1, epochs=1, shuffle=False)

    with tf.name_scope('inference'):
      inferences = self.build_inference(inputs, training=False)

    with tf.name_scope('output'):
      outputs = self.build_output(inferences)

    with tf.name_scope('evaluation'):
      metric, eval_op = self.build_evaluation(inputs, outputs)

    with tf.name_scope('initialization'):
      # Create the saver that will be used to restore trained variables,
      saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

      init_op, local_init_op = self.build_init()

    # Create the summary op that will merge all summaries across all sub-graphs
    summary_op = tf.summary.merge_all()

    return {
      'metric': metric,
      'init_op': init_op,
      'local_init_op': local_init_op,
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
      inputs = self.build_input(source=None, batch=0, epochs=0, shuffle=False)

    with tf.name_scope('inference'):
      inferences = self.build_inference(inputs, training=False)

    with tf.name_scope('output'):
      outputs = self.build_output(inferences)

    with tf.name_scope('initialization'):
      # Create the saver that will be used to restore trained variables.
      saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

      init_op, local_init_op = self.build_init()

    graph_inputs = tf.get_collection('inputs')
    if len(graph_inputs) != 1 or graph_inputs[0].dtype != tf.string:
      raise Exception('Invalid prediction graph. Must have a single string input.')

    graph_outputs = tf.get_collection('outputs')
    if len(graph_outputs) == 0:
      raise Exception('Invalid prediction graph. Must have at least one output.')

    return {
      'init_op': init_op,
      'local_init_op': local_init_op,
      'saver': saver,
      'inputs': graph_inputs,
      'outputs': graph_outputs
    }

  def build_init(self):
    """Builds the initialization sub-graph.

    The default implementation creates an initialization op that initializes all variables,
    locals for initialization, and another for all non-traininable variables and tables for local
    initialization.

    Initialization is run when the graph is first created, before training. Local initialization is
    performed after a previously trained model is loaded.

    Returns:
      A tuple containing the init op and local init op to use to initialize the graph.
    """
    init_op = tf.variables_initializer(tf.global_variables(), name='init')

    # For some reason not all local variables are in the local variables collection, but some are in
    # the global variables collection (such as those setup by reader ops).
    # So in addition to initializing local variables in the local_init_op, we also initialize the
    # set of variables in the global variables, that are not trainable.
    # Just to add to the mix, tables are neither, and so must be explicitly included as well.
    # All of these will be initialized after restoring from a checkpoint.
    variables = tf.global_variables()
    for trainable in tf.trainable_variables():
      variables.remove(trainable)

    local_init_op = tf.group(tf.variables_initializer(variables),
                             tf.variables_initializer(tf.local_variables()),
                             tf.tables_initializer(),
                             name='local_init_op')
    tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, local_init_op)

    return init_op, local_init_op

  def build_input(self, source, batch, epochs, shuffle):
    """Builds the input sub-graph.

    Arguments:
      source: the name of data source to use for input (for training and evaluation).
      batch: the number of instances to read per batch.
      epochs: the number of passes over the data.
      shuffle: whether to shuffle the data.
    Returns:
      A dictionary of tensors key'ed by feature names.
    """
    prediction = False
    if source:
      with tf.name_scope('read'):
        instances = self._dataset[source].read(batch=batch, shuffle=shuffle, epochs=epochs)
    else:
      prediction = True
      instances = tf.placeholder(dtype=tf.string, shape=(None,), name='instances')
      tf.add_to_collection('inputs', instances)

    with tf.name_scope('parse'):
      parsed_instances = self._dataset.parse_instances(instances, prediction)

    if self._dataset.features:
      with tf.name_scope('transform'):
        transformer = tfx.data.Transformer(self._dataset)
        return transformer.transform(parsed_instances)
    else:
      return parsed_instances

  def build_inference(self, inputs, training):
    """Builds the inference sub-graph.

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
      training: whether the inference sub-graph is being built for the training graph.
    Returns:
      The inference values.
    """
    raise NotImplementedError('build_inference must be implemented in a derived class.')

  def build_training(self, global_steps, inputs, inferences):
    """Builds the training sub-graph.

    Arguments:
      global_steps: the global steps variable to use.
      inputs: the dictionary of tensors corresponding to the input.
      inferences: the inference values.
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

  def build_evaluation(self, inputs, outputs):
    """Builds the evaluation graph.abs

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
      outputs: the dictionary containing output tensors.
    Returns:
      The eval metric tensor and the eval op.
    """
    raise NotImplementedError('build_evaluation must be implemented in a derived class.')
