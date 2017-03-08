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
  if field.type != SchemaFieldType.numeric:
    raise ValueError('A log transform cannot be applied to non-numerical field "%s".' %
                     feature.field)

  # Add 1 to avoid log of 0 (still assuming the field does not have negative values)
  return tf.log(instances[feature.field] + 1, name='log')


def _scale(instances, feature, schema, metadata):
  """Applies the scale transform to a numeric field.
  """
  field = schema[feature.field]
  if field.type != SchemaFieldType.numeric:
    raise ValueError('A scale transform cannot be applied to non-numerical field "%s".' %
                     feature.field)

  transform = feature.transform
  md = metadata[feature.field]

  value = instances[feature.field]

  range_min = float(md['min'])
  range_max = float(md['max'])
  value = (value - range_min) / (range_max - range_min)

  if transform:
    target_min = float(transform['min'])
    target_max = float(transform['max'])
    if (target_min != 0.0) or (target_max != 1.0):
      value = value * (target_max - target_min) + target_min

  return tf.identity(value, name='scale')


def _bucketize(instances, feature, schema, metadata):
  """Applies the bucketize transform to a numeric field.
  """
  field = schema[feature.field]
  if field.type != SchemaFieldType.numeric:
    raise ValueError('A scale transform cannot be applied to non-numerical field "%s".' %
                     feature.field)

  transform = feature.transform
  boundaries = map(float, transform['boundaries'].split(','))

  # TODO: Figure out how to use tf.case instead of this contrib op
  from tensorflow.contrib.layers.python.ops.bucketization_op import bucketize

  # Create a one-hot encoded tensor. The dimension of this tensor is the set of buckets defined
  # by N boundaries == N + 1.
  # A squeeze is needed to remove the extra dimension added to the shape.
  value = instances[feature.field]

  value = tf.squeeze(tf.one_hot(bucketize(value, boundaries, name='bucket'),
                                depth=len(boundaries) + 1, on_value=1.0, off_value=0.0,
                                name='one_hot'),
                     axis=1, name='bucketize')
  value.set_shape((None, len(boundaries) + 1))
  return value


def _one_hot(instances, feature, schema, metadata):
  """Applies the one-hot transform to a discrete field.
  """
  field = schema[feature.field]
  if field.type != SchemaFieldType.discrete:
    raise ValueError('A one-hot transform cannot be applied to non-discrete field "%s".' %
                     feature.field)

  md = metadata[feature.field]
  if not md:
    raise ValueError('A one-hot transform requires metadata listing the unique values.')

  entries = md['entries']
  table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(entries,
                                                tf.range(0, len(entries), dtype=tf.int64),
                                                tf.string, tf.int64),
    default_value=len(entries), name='entries')

  # Create a one-hot encoded tensor with one added to the number of values to account for the
  # default value returned by the table for unknown/failed lookups.
  # A squeeze is needed to remove the extra dimension added to the shape.
  value = instances[feature.field]

  value = tf.squeeze(tf.one_hot(table.lookup(value), len(entries) + 1, on_value=1.0, off_value=0.0),
                     axis=1,
                     name='one_hot')
  value.set_shape((None, len(entries) + 1))
  return value


_transformers = {
    FeatureType.identity.name: _identity,
    FeatureType.target.name: _target,
    FeatureType.concat.name: _concat,
    FeatureType.log.name: _log,
    FeatureType.scale.name: _scale,
    FeatureType.bucketize.name: _bucketize,
    FeatureType.one_hot.name: _one_hot
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
