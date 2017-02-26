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

# _dataset.py
# Implementation of DataSet and DataSource classes.

import tensorflow as tf
from tensorflow.python.lib.io import file_io as tfio
from ._schema import Schema
from ._metadata import Metadata
from ._features import FeatureSet


class DataSet(object):
  """A class representing data to be used within a job.

  A DataSet contains one or more DataSource instances, each associated with a name.
  """
  def __init__(self, datasources, schema, metadata, features):
    """Initializes a DataSet with the specified DataSource instances.

    Arguments:
      datasources: the set of contained DataSource instances key'ed by name.
      schema: the description of the source data.
      metadata: additional per-field information associated with the data.
      features: the optional description of the transformed data.
    """
    self._datasources = datasources

    if type(schema) is str:
      # Interpret this as a file path if the value is a string
      schema = tfio.read_file_to_string(schema)
      schema = Schema.parse(schema)
    self._schema = schema

    if metadata:
      if type(metadata) is str:
        # Interpret this as a file path if the value is a string
        metadata = tfio.read_file_to_string(metadata)
        metadata = Metadata.parse(metadata)
    self._metadata = metadata

    if features:
      if type(features) is str:
        # Interpret this as a file path if the value is a string
        features = tfio.read_file_to_string(features)
        features = FeatureSet.parse(features)
    self._features = features

  @property
  def schema(self):
    """Retrives the schema associated with the DataSet.
    """
    return self._schema

  @property
  def metadata(self):
    """Retrives the metadata associated with the DataSet.
    """
    return self._metadata

  @property
  def features(self):
    """Retrives the features defined with the DataSet.
    """
    return self._features

  @property
  def sources(self):
    """Retrieves the names of the contained DataSource instances.
    """
    return self._datasources.keys()

  def __getitem__(self, index):
    """Retrieves a named DataSource within the DataSet.

    Arguments:
      index: the name of the DataSource to retrieve.
    Returns:
      The DataSource if there is one with the specified name; None otherwise.
    """
    return self._datasources.get(index, None)

  def __len__(self):
    """Retrieves the number of contained DataSource instances.
    """
    return len(self._datasources)

  def parse_instances(self, instances, prediction=False):
    """Parses input instances according to the associated schema, metadata and features.

    Arguments:
      instances: The tensor containing input strings.
      prediction: Whether the instances are being parsed for producing predictions or not.
    Returns:
      A dictionary of tensors key'ed by feature names.
    """
    raise NotImplementedError()


class DataSource(object):
  """A base class representing data that can be read for use in a job.
  """
  def __init__(self):
    """Initializes an instance of a DataSource.
    """
    pass

  def read(self, batch=128, shuffle=False, shuffle_buffer=1000, epochs=0, threads=1):
    """Reads the data represented by this DataSource using a TensorFlow reader.

    Arguments:
      batch: The number of records to read at a time.
      shuffle: Whether to shuffle the list of files.
      shuffle_buffer: When shuffling, the number of extra items to keep in the queue for randomness.
      epochs: The number of epochs or passes over the data to perform.
      threads: the number of threads to use to read from the queue.
    Returns:
      A tensor containing a list of instances read.
    """
    instances = self.read_instances(batch, shuffle, epochs)

    queue_capacity = (threads + 3) * batch
    if shuffle:
      queue_capacity = queue_capacity + shuffle_buffer
      return tf.train.shuffle_batch([instances],
                                    batch_size=batch, allow_smaller_final_batch=True,
                                    enqueue_many=True,
                                    capacity=queue_capacity,
                                    min_after_dequeue=shuffle_buffer,
                                    num_threads=threads,
                                    name='shuffle_batch')
    else:
      return tf.train.batch([instances], batch_size=batch, allow_smaller_final_batch=True,
                            enqueue_many=True, capacity=queue_capacity,
                            num_threads=threads,
                            name='batch')

  def read_instances(self, count, shuffle, epochs):
    """Reads the data represented by this DataSource using a TensorFlow reader.

    Arguments:
      count: The number of instances to read in at most.
      shuffle: Whether to shuffle the input queue of files.
      epochs: The number of epochs or passes over the data to perform.
    Returns:
      A tensor containing instances that are read.
    """
    raise NotImplementedError('read_instances must be implemented in a derived class.')
