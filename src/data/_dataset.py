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


class DataSet(object):
  """A class representing data to be used within a job.

  A DataSet contains one or more DataSource instances, each associated with a name.
  """
  def __init__(self, datasources):
    """Initializes a DataSet with the specified DataSource instances.

    Arguments:
      datasources: a set of named DataSource instances.
    """
    self._datasources = datasources

  @classmethod
  def create(cls, *args):
    """Creates a DataSet with the specified DataSource instances.

    Arguments:
      args: A list of named DataSource instances.
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
    return cls(datasources)

  def __getitem__(self, index):
    """Retrieves a named DataSource within the DataSet.

    Arguments:
      index: the name of the DataSource to retrieve.
    Returns:
      The DataSource if there is one with the specified name; None otherwise.
    """
    return self._datasources.get(index, None)


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

