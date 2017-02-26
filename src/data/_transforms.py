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

    tensors = {}
    if features.target:
      tensors['targets'] = tf.identity(instances[features.target.field], name='target')

    field_list = map(lambda f: instances[f.field], features)
    tensors['features'] = tf.transpose(tf.stack(field_list), name='features')

    return tensors
