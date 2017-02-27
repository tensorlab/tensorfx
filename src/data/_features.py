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

# _features.py
# Implementation of FeatureSet and related class.

import enum
import tensorflow as tf
import yaml


class FeatureType(enum.Enum):
  """Defines the type of Feature instances.
  """
  identity = 'identity'
  target = 'target'
  concat = 'concat'
  log = 'log'
  scale = 'scale'
  bucketize = 'bucketize'
  one_hot = 'one-hot'


def _lookup_feature_type(s):
  for t in FeatureType:
    if t.value == s:
      return t
  raise ValueError('Invalid FeatureType "%s".' % s)


class Feature(object):
  """Defines a named feature within a FeatureSet.
  """
  def __init__(self, name, type, fields=None, features=None, transform=None):
    """Initializes a Feature with its name and source fields.

    Arguments:
      name: the name of the feature.
      type: the type of the feature.
      fields: the names of the fields making up this feature.
      features: the names of the features making up this feature in case of composite features.
      transform: transform configuration to produce the feature.
    """
    self._name = name
    self._type = type
    self._fields = fields
    self._features = features
    self._transform = transform

  @classmethod
  def identity(cls, name, field):
    """Creates a feature representing an un-transformed schema field.

    Arguments:
      name: the name of the feature.
      field: the name of the field.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.identity, fields=[field])

  @classmethod
  def target(cls, name, field):
    """Creates a feature representing the target value.
    
    Arguments:
      name: the name of the feature.
      field: the name of the field.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.target, fields=[field])

  @classmethod
  def concatenate(cls, name, *args):
    """Creates a composite feature that is a concatenation of multiple features.

    Arguments:
      name: the name of the feature.
      args: the sequence of features to concatenate.
    Returns:
      An instance of a Feature.
    """
    if not len(args):
      raise ValueError('One or more features must be specified.')

    if type(args[0]) == list:
      features = args[0]
    else:
      features = list(args)

    return cls(name, FeatureType.concat, features=features)

  @classmethod
  def log(cls, name, field):
    """Creates a feature representing a log value of a numeric field.

    Arguments:
      name: The name of the feature.
      field: The name of the field to create the feature from.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.log, fields=[field])

  @classmethod
  def scale(cls, name, field, range=(0, 1)):
    """Creates a feature representing a scaled version of a numeric field.

    In order to perform scaling, the metadata will be looked up for the field, to retrieve min, max
    and mean values.

    Arguments:
      name: The name of the feature.
      field: The name of the field to create the feature from.
      range: The target range of the feature.
    Returns:
      An instance of a Feature.
    """
    # TODO: What about the other scaling approaches, besides this (min-max scaling)?
    transform = {'min': range[0], 'max': range[1]}
    return cls(name, FeatureType.scale, fields=[field], transform=transform)

  @classmethod
  def bucketize(cls, name, field, boundaries):
    """Creates a feature representing a bucketized version of a numeric field.

    The value is returned is the index of the bucket that the value falls into in one-hot
    representation.

    Arguments:
      name: The name of the feature.
      field: The name of the field to create the feature from.
      boundaries: The list of bucket boundaries.
    Returns:
      An instance of a Feature.
    """
    transform = {'boundaries': ','.join(map(str, boundaries))}
    return cls(name, FeatureType.bucketize, fields=[field], transform=transform)

  @classmethod
  def one_hot(cls, name, field):
    """Creates a feature representing a one-hot representation of a discrete field.

    Arguments:
      name: The name of the feature.
      field: The name of the field to create the feature from.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.one_hot, fields=[field])

  @property
  def name(self):
    """Retrieves the name of the feature.
    """
    return self._name

  @property
  def features(self):
    """Retrieves the features making up a composite feature.
    """
    return self._features
  
  @property
  def field(self):
    """Retrieves the field making up the feature if the feature is based on a single field.
    """
    if len(self._fields) == 1:
      return self._fields[0]
    return None

  @property
  def fields(self):
    """Retrieves the fields making up the feature.
    """
    return self._fields
  
  @property
  def type(self):
    """Retrieves the type of the feature.
    """
    return self._type
  
  @property
  def transform(self):
    """Retrieves the transform configuration to produce the feature.
    """
    return self._transform

  def format(self):
    """Retrieves the raw serializable representation of the features.
    """
    data = {'name': self._name, 'type': self._type.value}
    if self._fields:
      data['fields'] = ','.join(self._fields)
    if self._transform:
      data['transform'] = self._transform
    if self._features:
      data['features'] = map(lambda f: f.format(), self._features)
    return data

  @staticmethod
  def parse(data):
    """Parses a feature from its serialized data representation.

    Arguments:
      data: A dictionary holding the serialized representation.
    Returns:
      The parsed Feature instance.
    """
    name = data['name']
    feature_type = _lookup_feature_type(data.get('type', 'identity'))
    transform = data.get('transform', None)

    fields = None
    features = None
    if feature_type == FeatureType.concat:
      features = []
      for f in data['features']:
        feature = Feature.parse(f)
        features.append(feature)
    else:
      fields = data.get('fields', name)
      if type(fields) is str:
        fields = map(lambda n: n.strip(), fields.split(','))

    return Feature(name, feature_type, fields=fields, features=features, transform=transform)


class FeatureSet(object):
  """Represents the set of features consumed by a model during training and prediction.

  A FeatureSet contains a set of named features. Features are derived from input fields specified
  in a schema and constructed using a transformation.
  """
  def __init__(self, features):
    """Initializes a FeatureSet from its specified set of features.

    Arguments:
      features: the list of features within a FeatureSet.
    """
    self._features = features
    self._features_map = dict(map(lambda f: (f.name, f), features))

  @staticmethod
  def create(*args):
    """Creates a FeatureSet from a set of features.

    Arguments:
      args: a list or sequence of features defining the FeatureSet.
    Returns:
      A FeatureSet instance.
    """
    if not len(args):
      raise ValueError('One or more features must be specified.')

    if type(args[0]) == list:
      return FeatureSet(args[0])
    else:
      return FeatureSet(list(args))

  @staticmethod
  def parse(spec):
    """Parses a FeatureSet from a YAML specification.

    Arguments:
      spec: The feature specification to parse.
    Returns:
      A FeatureSet instance.
    """
    if isinstance(spec, FeatureSet):
      return spec

    spec = yaml.safe_load(spec)

    features = []
    for f in spec['features']:
      feature = Feature.parse(f)
      features.append(feature)

    return FeatureSet(features)

  def __getitem__(self, index):
    """Retrives the specified Feature by name.

    Arguments:
      index: the name of the feature.
    Returns:
      The SchemaField if it exists; None otherwise.
    """
    return self._features_map.get(index, None)

  def __len__(self):
    """Retrieves the number of Features defined.
    """
    return len(self._features)

  def __iter__(self):
    """Creates an iterator over the features in the FeatureSet.
    """
    for feature in self._features:
      yield feature
