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

# _ds_df.py
# Implementation of DataFrameDataSet and DataFrameDataSource.

import numpy as np
import tensorflow as tf
from ._dataset import DataSet, DataSource
from ._schema import Schema, SchemaField, SchemaFieldType
from ._ds_csv import parse_csv


class DataFrameDataSet(DataSet):
  """A DataSet representing data loaded as Pandas DataFrame instances.
  """
  def __init__(self, features=None, **kwargs):
    """Initializes a DataFrameDataSet with the specified DataSource instances.

    Arguments:
      features: the optional description of the transformed data.
      kwargs: the set of CsvDataSource instances or csv paths to populate this DataSet with.
    """
    # Import pandas here, rather than always, to restrict loading the library at startup, as well as
    # having only a soft-dependency on the library.
    # Since the user is passing in DataFrame instances, the assumption is the library has been
    # loaded, and can be assumed to be installed.
    import pandas as pd

    def create_schema(df):
      fields = []
      for name, dtype in zip(df.columns, df.dtypes):
        if type(dtype) == pd.types.dtypes.CategoricalDtype:
          fields.append(SchemaField.discrete(name))
        elif dtype in (np.int32, np.int64, np.float32, np.float64):
          fields.append(SchemaField.numeric(name))
        else:
          raise ValueError('Unsupported data type "%s" in column "%s"' % (str(dtype), name))
      return Schema(fields)

    def create_metadata(df):
      metadata = {}
      for name, dtype in zip(df.columns, df.dtypes):
        md = {}
        if type(dtype) == pd.types.dtypes.CategoricalDtype:
          entries = list(df[name].unique())
          if np.nan in entries:
            entries.remove(np.nan)
          md['entries'] = sorted(entries)
        elif dtype in (np.int32, np.int64, np.float32, np.float64):
          for stat, stat_value in df[name].describe().iteritems():
            if stat == 'min':
              md['min'] = stat_value
            if stat == 'max':
              md['max'] = stat_value
        metadata[name] = md
      return metadata

    schema = None
    metadata = None
    datasources = {}
    for name, value in kwargs.iteritems():
      if isinstance(value, pd.DataFrame):
        value = DataFrameDataSource(value)

      if isinstance(value, DataFrameDataSource):
        datasources[name] = value
      else:
        raise ValueError('The specified DataSource is not a DataFrameDataSource')

      if not schema:
        schema = create_schema(value.dataframe)
      if not metadata:
        metadata = create_metadata(value.dataframe)

    if not len(datasources):
      raise ValueError('At least one DataSource must be specified.')

    super(DataFrameDataSet, self).__init__(datasources, schema, metadata, features)

  def parse_instances(self, instances, prediction=False):
    """Parses input instances according to the associated schema.

    Arguments:
      instances: The tensor containing input strings.
      prediction: Whether the instances are being parsed for producing predictions or not.
    Returns:
      A dictionary of tensors key'ed by feature names.
    """
    return parse_csv(self.schema, instances, prediction)


class DataFrameDataSource(DataSource):
  """A DataSource representing a Pandas DataFrame.

  This class is useful for working with local/in-memory data.
  """
  def __init__(self, df):
    """Initializes an instance of a DataFrameDataSource with the specified Pandas DataFrame.

    Arguments:
      df: the DataFrame instance to use.
    """
    super(DataFrameDataSource, self).__init__()
    self._df = df
  
  @property
  def dataframe(self):
    """Retrieves the DataFrame represented by this DataSource.
    """
    return self._df

  def read_instances(self, count, shuffle, epochs):
    """Reads the data represented by this DataSource using a TensorFlow reader.

    Arguments:
      epochs: The number of epochs or passes over the data to perform.
    Returns:
      A tensor containing instances that are read.
    """
    # None implies unlimited; switch the value to None when epochs is 0.
    epochs = epochs or None

    with tf.device(''):
      # Ensure the device is local and the queue, dequeuing and lookup all happen on the default
      # device, which is required for the py_func operation.

      # A UDF that given a batch of indices, returns a batch of string (csv formatted) instances
      # from the DataFrame.
      df = self._df
      def reader(indices):
        rows = df.iloc[indices]
        return [map(lambda r: ','.join(r), rows.values.astype('string'))]

      queue = tf.train.range_input_producer(self._df.shape[0], num_epochs=epochs, shuffle=shuffle,
                                            name='queue')
      indices = queue.dequeue_up_to(count)
      instances = tf.py_func(reader, [indices], tf.string, name='read')
      instances.set_shape((None,))

    return instances
