# Copyright 2016 TensorLabs. All rights reserved.
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

# schema_tests.py
# Tests FeatureSet related functionality in tensorfx.data

import unittest
import tensorfx as tfx


class TestCases(unittest.TestCase):

  def test_create_featureset(self):
    t = tfx.data.Feature.target('t', 't')
    x = tfx.data.Feature.identity('x', 'x')
    features = tfx.data.FeatureSet.create(t, x)

    self.assertEqual(len(features), 2)
    self.assertEqual(features['t'], t)

  def test_parse_featureset(self):
    spec = """
    features:
    - name: target
      type: target
      fields: c1
    - name: f1
      type: identity
      fields: c3
    """
    features = tfx.data.FeatureSet.parse(spec)

    self.assertEqual(len(features), 2)
    self.assertEqual(features['target'].fields[0], 'c1')
    self.assertEqual(features['target'].type, tfx.data.FeatureType.target)
    self.assertEqual(features['f1'].type, tfx.data.FeatureType.identity)
    self.assertEqual(features['f1'].fields, ['c3'])
