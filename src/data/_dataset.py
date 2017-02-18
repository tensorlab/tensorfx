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
  def __init__(self, datasources, schema, metadata=None, features=None):
    """Initializes a DataSet with the specified DataSource instances.

    Arguments:
      datasources: a set of named DataSource instances.
      schema: the description of the source data.
      metadata: additional per-field information associated with the data.
      features: the optional description of the transformed data.
    """
    self._datasources = datasources
    self._schema = schema
    self._metadata = metadata
    self._features = features

  @classmethod
  def create(cls, schema, *args, **kwargs):
    """Creates a DataSet with the specified DataSource instances.

    Arguments:
      schema: the description of the source data.
      args: A list of named DataSource instances.
      kwargs: optional information, such as metadata and features.
    Returns:
      A DataSet containing the specified DataSource instances.
    Raises:
      ValueError if the list of DataSources is empty, or not a homogeneous set of instances.
    """
    if not len(args):
      raise ValueError('One or more DataSource instances must be specified.')
    if not isinstance(args[0], DataSource) or \
       not all(map(lambda ds: type(ds) == type(args[0]), args)):
      raise ValueError('All the listed DataSource instances must be of the same type.')

    datasources = dict(map(lambda ds: (ds.name, ds), args))
    return cls(datasources, schema,
               kwargs.get('metadata', None),
               kwargs.get('features', None))

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

  @staticmethod
  def parse(spec, schema, **kwargs):
    """Creates a DataSet with the specified set of DataSource urls.

    Arguments:
      spec: A specification of the DataSet.
      schema: the description of the source data.
      kwargs: optional information, such as metadata and features.
    Returns:
      A DataSet containing the specified DataSource instances.
    Raises:
      ValueError if the list of DataSources is empty, not-parseable or heterogenous.
    """
    # TODO: Should the strings be interpretted as file paths (as they are), or file contents?
    if type(schema) is str:
      # Interpret this as a file path if the value is a string
      schema = tfio.read_file_to_string(schema)
      schema = Schema.parse(schema)

    metadata = kwargs.get('metadata', None)
    if metadata:
      if type(metadata) is str:
        # Interpret this as a file path if the value is a string
        metadata = tfio.read_file_to_string(metadata)
        metadata = Metadata.parse(metadata)

    features = kwargs.get('features', None)
    if features:
      if type(features) is str:
        # Interpret this as a file path if the value is a string
        features = tfio.read_file_to_string(features)
        features = FeatureSet.parse(features)

    data_format = spec['format']
    dataset_type = DataSetRegistry.lookup(data_format)
    data_sources = []
    for name, path in spec['sources'].iteritems():
      data_sources.append(dataset_type.create_datasource(data_format, name, path))

    return dataset_type.create(schema, *data_sources, metadata=metadata, features=features)

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
  def __init__(self, name):
    """Initializes an instance of a DataSource.

    Arguments:
      name: the name of the DataSource.
    """
    self._name = name

  @property
  def name(self):
    """Retrieves the name of the DataSource.
    """
    return self._name

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


class DataSetRegistry(object):
  """Implements a registry of dataset formats to dataset types.
  """
  _mapping = dict()

  @staticmethod
  def lookup(format):
    """Looks up a dataset by a registered format name.

    Arguments:
      type: the type of dataset to lookup.
    Returns:
      The type of DataSet identified by the format, or None if not found.
    """
    return DataSetRegistry._mapping.get(format, None)

  @staticmethod
  def register(format, type):
    """Registers a DataSet type.

    The DataSet type must have a static create_datasource method that accepts a format, name, and
    path and returns a DataSource instance.

    Arguments:
      format: the data format to associate this DataSet type with.
      type: the data source type.
    """
    DataSetRegistry._mapping[format] = type
