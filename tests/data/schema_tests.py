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
# Tests Schema related functionality in tensorfx.data

import unittest
import tensorfx as tfx


class TestCases(unittest.TestCase):

  def test_create_single_field_schema(self):
    f = tfx.data.SchemaField.numeric('n')
    schema = tfx.data.Schema.create(f)

    self.assertEqual(len(schema), 1)
    self.assertEqual(schema['n'], f)
    self.assertEqual(schema[0], f)

  def test_create_multi_field_schema(self):
    f1 = tfx.data.SchemaField.numeric('n')
    f2 = tfx.data.SchemaField.text('t')
    schema = tfx.data.Schema.create(f1, f2)

    self.assertEqual(len(schema), 2)
    self.assertEqual(schema['n'], f1)
    self.assertEqual(schema[1], f2)

  def test_parse_schema(self):
    spec = """
    fields:
    - name: f1
      type: numeric
    - name: f2
      type: text
    - name: f3
      type: discrete
    """
    schema = tfx.data.Schema.parse(spec)

    self.assertEqual(len(schema), 3)
    self.assertEqual(schema[0].name, 'f1')
    self.assertEqual(schema['f1'].type, tfx.data.SchemaFieldType.numeric)
    self.assertEqual(schema.fields, ['f1', 'f2', 'f3'])
  