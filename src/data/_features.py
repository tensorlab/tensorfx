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

import yaml


class Feature(object):
  """Defines a named feature within a FeatureSet.
  """
  def __init__(self, name, fields):
    """Initializes a Feature with its name and source fields.

    Arguments:
      name: the name of the feature.
      fields: the names of the fields making up this feature.
    """
    self._name = name
    self._fields = fields
  
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

  @classmethod
  def parse(cls, spec):
    """Parses a FeatureSet from a YAML specification.

    Arguments:
      spec: The feature specification to parse.
    Returns:
      A FeatureSet instance.
    """
    # TODO: Implement this
    return None

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
