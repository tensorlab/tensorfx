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

# dataset_tests.py
# Tests Dataset related functionality in tensorfx.data.

import unittest
import tensorfx as tfx


class TestCases(unittest.TestCase):

  def test_empty_dataset(self):
    schema = tfx.data.Schema.create(tfx.data.SchemaField.numeric('x'))
    ds = tfx.data.DataSet({}, schema, None, None)

    self.assertEqual(len(ds), 0)

  def test_create_dataset(self):
    schema = tfx.data.Schema.create(tfx.data.SchemaField.numeric('x'))
    source = tfx.data.DataSource()
    ds = tfx.data.DataSet({'foo': source}, schema, None, None)

    self.assertEqual(ds['foo'], source)

  def test_create_multi_source_dataset(self):
    schema = tfx.data.Schema.create(tfx.data.SchemaField.numeric('x'),
                                    tfx.data.SchemaField.numeric('y'))
    train = tfx.data.CsvDataSource('...')
    eval = tfx.data.CsvDataSource('...')

    ds = tfx.data.CsvDataSet(schema, train=train, eval=eval)

    self.assertEqual(ds['train'], train)
    self.assertEqual(ds['eval'], eval)
    self.assertEqual(len(ds), 2)
    self.assertListEqual(ds.sources, ['train', 'eval'])
