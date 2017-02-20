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
# Implements ClassificationScenario.

import tensorflow as tf

class ClassificationScenario(object):
  """Represents a classification machine learning scenario.
  """
  def __init__(self, labels):
    """Initializes an instance of a Classification object.

    Arguments:
      labels: the ordered list of strings representing the labels being predicted.
    """
    self._labels = labels

  @property
  def labels(self):
    """Retrieves the labels being predicted.
    """
    return self._labels

  @property
  def num_labels(self):
    """Retrieves the number of labels being predicted.
    """
    return len(self._labels)

  def labels_to_indices(self, strings, one_hot, name='indices'):
    """Converts the specified string label names into equivalent indices.

    The implementation builds a HashTable, to perform a lookup operation.

    Arguments:
      strings: the string labels to convert into equivalent indices.
      one_hot: whether to convert the indices into their one-hot representation.
    Returns:
      A set of tensors representing the labels by indices.
    """
    num_labels = len(self._labels)
    with tf.name_scope('label_table'):
      table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(self._labels,
                                                    tf.range(0, num_labels, dtype=tf.int64),
                                                    tf.string, tf.int64), -1)
    if one_hot:
      indices = tf.squeeze(tf.one_hot(table.lookup(strings), num_labels), name=name)
    else:
      indices = table.lookup(strings, name=name)
    
    return indices

  def indices_to_labels(self, indices, name='label'):
    """Converts the specified integer indices into equivalent string label names.

    The implementation builds a HashTable, to perform a lookup operation.

    Arguments:
      indices: the int64 indices to convert into equivalent strings.
    Returns:
      A set of tensors representing the labels by name.
    """
    num_labels = len(self._labels)
    with tf.name_scope('label_table'):
      table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(tf.range(0, num_labels, dtype=tf.int64),
                                                    self._labels,
                                                    tf.int64, tf.string), '')
    return table.lookup(indices, name=name)
