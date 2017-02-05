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
from ._dataset import DataSet, DataSource, DataSourceRegistry


class CsvDataSource(DataSource):
  """A DataSource representing one or more csv files.
  """
  def __init__(self, name, path, delimiter=',', header=False):
    """Initializes an instance of a CsvDataSource with the specified csv file(s).

    Arguments:
      name: the name of the DataSource.
      path: the csv file containing the data. This can be a pattern to represent a set of files.
      delimiter: the delimiter character used.
      header: whether the file contains a header line that should be skipped.
    """
    super(CsvDataSource, self).__init__(name)
    self._path = path
    self._delimiter = delimiter
    self._header = header

  @classmethod
  def create(cls, name, scheme, path):
    """Creates an instance of the DataSource from its spec.

    Arguments:
      name: the name of the DataSource.
      scheme: the scheme of the data parsed from a data source spec.
      path: the path of the data parsed from a data source spec.
    """
    if scheme == 'csv':
      return cls(name, path)
    elif schem == 'tsv':
      return cls(name, path, delimiter='\t')
    else:
      raise ValueError('Unknown scheme "%s"' % schema)

  @property
  def path(self):
    """Retrives the path represented by the DataSource.
    """
    return self._path

  def read_instances(self, epochs=0):
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
    reader = tf.TextLineReader(skip_header_lines=int(self._header), name='reader')
    _, instances = reader.read(queue, name='read')

    return instances


DataSourceRegistry.register('csv', CsvDataSource)
DataSourceRegistry.register('tsv', CsvDataSource)
