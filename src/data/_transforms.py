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

# _transforms.py
# Implementation of various transforms to build features.

import tensorflow as tf
from ._features import FeatureType
from ._schema import SchemaFieldType


class Transformer(object):
  """Implements transformation logic.
  """
  def __init__(self, dataset):
    """Initializes a Transformer.

    Arguments:
      dataset: The dataset containing the data to be transformed into features.
    """
    self._dataset = dataset

  def transform(self, instances):
    """Transforms the supplied instances into features.

    Arguments:
      instances: a dictionary of tensors key'ed by field names corresponding to the schema.
    Returns:
      A dictionary of tensors key'ed by feature names corresponding to the feature set.
    """
    features = self._dataset.features

    # The top-level set of features is to be represented as a map of tensors, so transform the
    # features, and use the map result.
    _, tensor_map = _transform_features(instances, features,
                                        self._dataset.schema,
                                        self._dataset.metadata)
    return tensor_map


def _identity(instances, feature, schema, metadata):
  """Applies the identity transform, which causes the unmodified field value to be used.
  """
  return tf.identity(instances[feature.field], name='identity')


def _target(instances, feature, schema, metadata):
  """Applies the target transform, which causes the unmodified field value to be used.
  """
  # The result of parsing csv is a tensor of shape (None, 1), and we want to return a list of
  # scalars, or specifically, tensor of shape (None, ).
  return tf.squeeze(instances[feature.field], name='target')


def _concat(instances, feature, schema, metadata):
  """Applies the composite transform, to compose a single tensor from a set of features.
  """
  tensors, _ = _transform_features(instances, feature.features, schema, metadata)
  return tf.concat(tensors, axis=1, name='concat')


def _log(instances, feature, schema, metadata):
  """Applies the log transform to a numeric field.
  """
  field = schema[feature.field]
  if field.type != SchemaFieldType.real and field.type != SchemaFieldType.integer:
    raise ValueError('A log transform cannot be applied to non-numerical field "%s".' %
                     feature.field)

  # log can only be applied to float, so do an implict conversion first.
  return tf.log(tf.to_float(instances[feature.field], name='float'), name='log')


def _scale(instances, feature, schema, metadata):
  """Applies the scale transform to a numeric field.
  """
  field = schema[feature.field]
  if field.type != SchemaFieldType.real and field.type != SchemaFieldType.integer:
    raise ValueError('A scale transform cannot be applied to non-numerical field "%s".' %
                     feature.field)

  transform = feature.transform
  md = metadata[feature.field]

  value = instances[feature.field]
  if transform and transform['log']:
    # log can only be applied to float, so do an implict conversion first.
    value = tf.log(tf.to_float(value, name='float'), name='log')

  range_min = float(md['min'])
  range_max = float(md['max'])
  value = (value - range_min) / (range_max - range_min)

  if transform:
    target_min = float(transform['min'])
    target_max = floag(transform['max'])
    if (target_min != 0.0) or (target_max != 1.0):
      value = value * (target_max - target_min) + target_min

  return tf.identity(value, name='scale')


_transformers = {
    FeatureType.identity.name: _identity,
    FeatureType.target.name: _target,
    FeatureType.concat.name: _concat,
    FeatureType.log.name: _log,
    FeatureType.scale.name: _scale
  }

def _transform_features(instances, features, schema, metadata):
  """Transforms a list of features, to produce a list and map of tensor values.
  """
  tensors = []
  tensor_map = {}

  for f in features:
    transformer = _transformers[f.type.name]
    with tf.name_scope(f.name):
      value = transformer(instances, f, schema, metadata)

    tensors.append(value)
    tensor_map[f.name] = value

  return tensors, tensor_map
