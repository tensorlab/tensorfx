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

# _schema.py
# Implementation of Schema and related classes.

import enum

class SchemaFieldType(enum.Enum):
  """Defines the types of SchemaField instances.
  """
  integer = 'integer'
  real = 'real'
  discrete = 'discrete'
  text = 'text'
  binary = 'binary'


class SchemaField(object):
  """Defines a named and typed field within a Schema.
  """
  def __init__(self, name, type):
    """Initializes a SchemaField with its name and type.

    Arguments:
      name: the name of the field.
      type: the type of the field.
    """
    self._name = name
    self._type = type

  @classmethod
  def integer(cls, name):
    """Creates a field representing an integer.

    Arguments:
      name: the name of the field.
    """
    return cls(name, SchemaFieldType.integer)

  @classmethod
  def real(cls, name):
    """Creates a field representing a real number.

    Arguments:
      name: the name of the field.
    """
    return cls(name, SchemaFieldType.real)

  @classmethod
  def discrete(cls, name):
    """Creates a field representing a discrete value.

    Arguments:
      name: the name of the field.
    """
    return cls(name, SchemaFieldType.discrete)

  @classmethod
  def text(cls, name):
    """Creates a field representing a text string.

    Arguments:
      name: the name of the field.
    """
    return cls(name, SchemaFieldType.text)

  @classmethod
  def binary(cls, name):
    """Creates a field representing a binary byte buffer.

    Arguments:
      name: the name of the field.
    """
    return cls(name, SchemaFieldType.binary)

  @property
  def name(self):
    """Retrieves the name of the field.
    """
    return self._name
  
  @property
  def type(self):
    """Retrieves the type of the field.
    """
    return self._type


class Schema(object):
  """Defines the schema of a DataSet.

  The schema represents the structure of the source data before it is transformed into features.
  """
  def __init__(self, fields):
    """Initializes a Schema with the specified set of fields.

    Arguments:
      fields: a list of fields representing an ordered set of columns.
    """
    self._fields = fields
    self._field_set = dict(map(lambda f: (f.name, f), fields))

  @classmethod
  def create(cls, *args):
    """Creates a Schema from a set of fields.

    Arguments:
      args: a list or sequence of ordered fields defining the schema.
    """
    if not len(args):
      raise ValueError('One or more fields must be specified.')

    if type(args[0]) == list:
      return cls(args[0])
    else:
      return cls(list(args))

  def __getattr__(self, attr):
    """Retrieves the specified SchemaField by name.

    Arguments:
      attr: the name of the SchemaField to retrieve.
    Returns:
      The SchemaField with the specified name.
    Raises:
      AttributeError if the specified name is not found.
    """
    field = self._field_set.get(attr, None)
    if field is None:
      raise AttributeError
    return field

  def __getitem__(self, index):
    """Retrives the specified SchemaField by name or position.

    Arguments:
      index: the name or index of the field.
    Returns:
      The SchemaField if it exists; None otherwise.
    """
    if type(index) is int:
      return self._fields[index] if len(self._fields) > index else None
    else:
      return self._field_set.get(index, None)

  def __len__(self):
    """Retrieves the number of SchemaFields defined.
    """
    return len(self._fields)
