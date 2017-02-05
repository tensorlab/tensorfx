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
import yaml

class FeatureType(enum.Enum):
  """Defines the type of Feature instances.
  """
  identity = 'identity'
  target = 'target'
  key = 'key'


class Feature(object):
  """Defines a named feature within a FeatureSet.
  """
  def __init__(self, name, type, fields, transform=None):
    """Initializes a Feature with its name and source fields.

    Arguments:
      name: the name of the feature.
      type: the type of the feature.
      fields: the names of the fields making up this feature.
      transform: transform configuration to produce the feature.
    """
    if transform is None:
      transform = {}

    self._name = name
    self._type = type
    self._fields = fields
    self._transform = transform

  @classmethod
  def identity(cls, name, field):
    """Creates a feature representing an un-transformed schema field.

    Arguments:
      name: the name of the feature.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.identity, [field])

  @classmethod
  def key(cls, name, field):
    """Creates a feature representing an un-transformed schema field with key semantics.

    Key features are usually passthrough with respect to the model. They can be used to join
    input datasets and output predictions.

    Arguments:
      name: the name of the feature.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.key, [field])

  @classmethod
  def target(cls, name, field):
    """Creates a feature representing the target value.
    
    Arguments:
      name: the name of the feature.
    Returns:
      An instance of a Feature.
    """
    return cls(name, FeatureType.target, [field])
  
  @property
  def name(self):
    """Retrieves the name of the feature.
    """
    return self._name
  
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


class FeatureSet(object):
  """Represents the set of features consumed by a model during training and prediction.

  A FeatureSet contains a set of named features. Features are derived from input fields specified
  in a schema and constructed using a transformation.
  """
  def __init__(self, features):
    """Initializes a FeatureSet from its specified set of features.

    Arguments:
      features: the set of features within a FeatureSet.
    """
    self._features = features
    self._feature_map = dict(map(lambda f: (f.name, f), features))

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

    features = [
      Feature.target('@target', spec['target']),
      Feature.key('@key', spec['key'])
    ]
    for f in spec['features']:
      fields = f['fields']
      if type(fields) is str:
        fields = map(lambda n: n.strip(), fields.split(','))

      feature = Feature(f['name'], FeatureType[f['type']], fields,f.get('transform', None))
      features.append(feature)

    return FeatureSet(features)

  def __getitem__(self, index):
    """Retrives the specified SchemaField by name or position.

    Arguments:
      index: the name or index of the field.
    Returns:
      The SchemaField if it exists; None otherwise.
    """
    if type(index) is int:
      return self._features[index] if len(self._features) > index else None
    else:
      return self._feature_map.get(index, None)

  def __len__(self):
    """Retrieves the number of Features defined.
    """
    return len(self._features)
