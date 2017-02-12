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

# _ds_csv.py
# Implementation of CsvDataSource.

import tensorflow as tf
from ._dataset import DataSet, DataSource, DataSetRegistry
from ._schema import SchemaFieldType


class CsvDataSet(DataSet):
  """A DataSet representing data in csv format.
  """
  def __init__(self, datasources, schema, metadata=None, features=None, refs=None):
    """Initializes a DataSet with the specified DataSource instances.

    Arguments:
      datasources: a set of named DataSource instances.
      schema: the description of the source data.
      metadata: additional per-field information associated with the data.
      features: the optional description of the transformed data.
    """
    super(CsvDataSet, self).__init__(datasources, schema, metadata, features, refs)

  @staticmethod
  def create_datasource(format, name, path):
    """Creates a DataSource representing the specified data.

    Arguments:
      format: The type of DataSource to create.
      name: The name of the DataSource.
      path: The path of the data.
    Returns:
      A DataSource representing the specified data.
    """
    if format == 'csv':
      return CsvDataSource(name, path)
    elif format == 'tsv':
      return CsvDataSource(name, path, delimiter='\t')
    else:
      raise ValueError('Unexpected format "%s"' % format)

  def parse_instances(self, instances, prediction=False):
    """Parses input instances according to the associated schema, metadata and features.

    Arguments:
      instances: The tensor containing input strings.
      prediction: Whether the instances are being parsed for producing predictions or not.
    Returns:
      A tuple consisting of targets and a dictionary of tensors key'ed by feature names.
    """
    if prediction:
      # For training and evaluation data, the expectation is the target column is always present.
      # For prediction however, the target may or may not be present.
      # - In true prediction use-cases, the target is unknown and never present.
      # - In prediction for model evaluation use-cases, the target is present.
      # To use a single prediction graph, the missing target needs to be detected by comparing
      # number of columns in instances with number of columns defined in the schema. If there are
      # fewer columns, then prepend a ',' (with assumption that target is always the first column).
      #
      # To get the number of columns in instances, split on the ',' on the first instance, and use
      # the first dimension of the shape of the resulting substring values.
      columns = tf.shape(tf.string_split([instances[0]], delimiter=',').values)[0]
      instances = tf.cond(tf.less(columns, len(self.schema)),
                          lambda: tf.string_join([tf.constant(','), instances]),
                          lambda: instances)

    # Convert the schema into a set of tensor defaults, to be used for parsing csv data.
    defaults = []
    for field in self.schema:
      if field.type == SchemaFieldType.integer:
        field_default = tf.constant(0, dtype=tf.int64)
      elif field.type == SchemaFieldType.real:
        field_default = tf.constant(0.0, dtype=tf.float32)
      else:
        # discrete, text, binary
        field_default = tf.constant('', dtype=tf.string)
      defaults.append([field_default])

    values = tf.decode_csv(instances, defaults)

    # TODO: Factor in features and metadata
    targets = None
    features = {}
    for field, value in zip(self.schema, values):
      if targets is None:
        targets = value
      else:
        features[field.name] = value

    return targets, features


DataSetRegistry.register('csv', CsvDataSet)
DataSetRegistry.register('tsv', CsvDataSet)


class CsvDataSource(DataSource):
  """A DataSource representing one or more csv files.
  """
  def __init__(self, name, path, delimiter=','):
    """Initializes an instance of a CsvDataSource with the specified csv file(s).

    Arguments:
      name: the name of the DataSource.
      path: the csv file containing the data. This can be a pattern to represent a set of files.
      delimiter: the delimiter character used.
    """
    super(CsvDataSource, self).__init__(name)
    self._path = path
    self._delimiter = delimiter

  @property
  def path(self):
    """Retrives the path represented by the DataSource.
    """
    return self._path

  def read_instances(self, count, shuffle, epochs):
    """Reads the data represented by this DataSource using a TensorFlow reader.

    Arguments:
      epochs: The number of epochs or passes over the data to perform.
    Returns:
      A tensor containing instances that are read.
    """
    # None implies unlimited; switch the value to None when epoch is 0.
    epochs = epochs or None

    files = tf.train.match_filenames_once(self._path, name='files')
    queue = tf.train.string_input_producer(files, num_epochs=epochs, shuffle=shuffle,
                                           name='queue')
    reader = tf.TextLineReader(name='reader')
    _, instances = reader.read_up_to(queue, count, name='read')

    return instances

