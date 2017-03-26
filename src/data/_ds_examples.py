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

# _ds_examples.py
# Implementation of ExamplesDataSource.

import tensorflow as tf
from ._dataset import DataSet, DataSource
from ._schema import SchemaFieldType


class ExamplesDataSet(DataSet):
  """A DataSet representing data in tf.Example protobuf within a TFRecord format.
  """
  def __init__(self, schema, metadata=None, features=None, **kwargs):
    """Initializes a ExamplesDataSet with the specified DataSource instances.

    Arguments:
      schema: the description of the source data.
      metadata: additional per-field information associated with the data.
      features: the optional description of the transformed data.
      kwargs: the set of ExamplesDataSource instances or TFRecord paths to populate this DataSet.
    """
    datasources = {}
    for name, value in kwargs.iteritems():
      if isinstance(value, str):
        value = ExamplesDataSource(value)

      if isinstance(value, ExamplesDataSource):
        datasources[name] = value
      else:
        raise ValueError('The specified DataSource is not a ExamplesDataSource')

    if not len(datasources):
      raise ValueError('At least one DataSource must be specified.')

    super(ExamplesDataSet, self).__init__(datasources, schema, metadata, features)

  def parse_instances(self, instances, prediction=False):
    """Parses input instances according to the associated schema.

    Arguments:
      instances: The tensor containing input strings.
      prediction: Whether the instances are being parsed for producing predictions or not.
    Returns:
      A dictionary of tensors key'ed by field names.
    """
    # Convert the schema into an equivalent Example schema (expressed as features in Example
    # terminology).
    features = {}
    for field in self.schema:
      if field.type == SchemaFieldType.integer:
        dtype = tf.int64
        default_value = [0]
      elif field.type == SchemaFieldType.real:
        dtype = tf.float32
        default_value = [0.0]
      else:
        # discrete
        dtype = tf.string
        default_value = ['']

      if field.length == 0:
        feature = tf.VarLenFeature(dtype=dtype)
      else:
        if field.length != 1:
          default_value = default_value * field.length
        feature = tf.FixedLenFeature(shape=[field.length], dtype=dtype, default_value=default_value)

      features[field.name] = feature

    return tf.parse_example(instances, features, name='examples')


class ExamplesDataSource(DataSource):
  """A DataSource representing one or more TFRecord files containing tf.Example data.
  """
  def __init__(self, path, compressed=False):
    """Initializes an instance of a ExamplesDataSource with the specified TFRecord file(s).

    Arguments:
      path: TFRecord file containing the data. This can be a pattern to represent a set of files.
      compressed: Whether the TFRecord files are compressed.
    """
    super(ExamplesDataSource, self).__init__()
    self._path = path
    self._compressed = compressed

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
    # None implies unlimited; switch the value to None when epochs is 0.
    epochs = epochs or None

    options = None
    if self._compressed:
      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    files = tf.train.match_filenames_once(self._path, name='files')
    queue = tf.train.string_input_producer(files, num_epochs=epochs, shuffle=shuffle,
                                           name='queue')
    reader = tf.TFRecordReader(options=options, name='reader')
    _, instances = reader.read_up_to(queue, count, name='read')

    return instances
