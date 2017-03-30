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

# _classification.py
# Implements ClassificationModelBuilder and ClassificationModelArguments.

import tensorflow as tf
import tensorfx as tfx

class ClassificationModelArguments(tfx.training.ModelArguments):
  """Arguments for classification models.
  """
  @classmethod
  def init_parser(cls, parser):
    """Initializes the argument parser.

    Args:
      parser: An argument parser instance to be initialized with arguments.
    """
    super(ClassificationModelArguments, cls).init_parser(parser)

  def process(self):
    """Processes the parsed arguments to produce any additional objects.
    """
    pass


class ClassificationModelBuilder(tfx.training.ModelBuilder):
  """A ModelBuilder for building classification models.

  A classification model treats the target value as a label. The label might be a discrete
  value (which is converted to integer indices), or may be pre-indexed.
  """
  def __init__(self, args):
    super(ClassificationModelBuilder, self).__init__(args)
    self._classification = None

  @property
  def classification(self):
    """Returns the classification helper object.
    """
    return self._classification

  def build_graph_interfaces(self, dataset, config):
    """Builds graph interfaces for training and evaluating a model, and for predicting using it.

    A graph interface is an object containing a TensorFlow graph member, as well as members
    corresponding to various tensors and ops within the graph.

    ClassificationModelBuilder also builds a classification helper object for use during graph
    building.

    Arguments:
      dataset: The dataset to use during training.
      config: The training Configuration object.
    Returns:
      A tuple consisting of the training, evaluation and prediction interfaces.
    """
    target_feature = filter(lambda f: f.type == tfx.data.FeatureType.target, dataset.features)[0]
    target_field = dataset.schema[target_feature.field]
    target_metadata = dataset.metadata[target_feature.field]

    if target_field.type == tfx.data.SchemaFieldType.discrete:
      self._classification = StringLabelClassification(target_metadata['vocab']['entries'])
    else:
      self._classification = None

    return super(ClassificationModelBuilder, self).build_graph_interfaces(dataset, config)


class StringLabelClassification(object):
  """A classification scenario involving string label names.

  Labels will be converted to indices when using the input, and indices back to labels to produce
  output.
  """
  def __init__(self, labels):
    """Initializes an instance of StringLabelClassification with specified label names.
    """
    self._labels = labels
    self._num_labels = len(labels)

  @property
  def num_labels(self):
    """Returns the number of labels in the model.
    """
    return self._num_labels

  def keys(self, inputs):
    """Retrieves the keys, if present from the inputs.

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
    Returns:
      A tensor containing the keys if a keys feature exists, None otherwise.
    """
    return inputs.get('key', None)

  def features(self, inputs):
    """Retrieves the features to use to build a model.

    For classification models, the default behavior is to use a feature named 'X' to represent the
    input features for the model.

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
    Returns:
      A tensor containing model input features.
    """
    return inputs['X']

  def target_labels(self, inputs):
    """Retrieves the target labels to use to build a model.

    For classification models, the default behavior is to use a feature named 'Y' to represent the
    target features for the model.

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
    Returns:
      A tensor containing the target labels.
    """
    return inputs['Y']

  def target_label_indices(self, inputs, one_hot=True):
    """Retrieves the target labels to use to build a model, as a set of indices.

    For classification models, the default behavior is to use a feature named 'Y' to represent the
    target features for the model. The labels are used to perform a lookup to produce indices.

    Arguments:
      inputs: the dictionary of tensors corresponding to the input.
      one_hot: whether to convert the indices into their one-hot representation.
    Returns:
      A tensor containing the target labels as indices..
    """
    labels = inputs['Y']

    with tf.name_scope('label_table'):
      string_int_mapping =  tf.contrib.lookup.KeyValueTensorInitializer(
        self._labels, tf.range(0, self._num_labels, dtype=tf.int64), tf.string, tf.int64)
      table = tf.contrib.lookup.HashTable(string_int_mapping, default_value=-1)

    if one_hot:
      indices = tf.squeeze(tf.one_hot(table.lookup(labels), self._num_labels), name='indices')
    else:
      indices = table.lookup(labels, name='indices')

    return indices

  def output_labels(self, indices):
    """Produces the output labels to represent a model's output.

    The indices are used to lookup corresponding label names.

    Arguments:
      indices: The predicted label indices.
    Returns:
      A tensor containing output predicted label names.
    """
    with tf.name_scope('label_table'):
      int_string_mapping = tf.contrib.lookup.KeyValueTensorInitializer(
        tf.range(0, self._num_labels, dtype=tf.int64), self._labels, tf.int64, tf.string)
      table = tf.contrib.lookup.HashTable(int_string_mapping, default_value='')

    return table.lookup(indices, name='label')
