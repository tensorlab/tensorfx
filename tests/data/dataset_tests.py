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

  def test_create_dataset(self):
    source = tfx.data.DataSource('foo')
    schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'))
    ds = tfx.data.DataSet.create(schema, source)

    self.assertEqual(ds['foo'], source)

  def test_create_multi_source_dataset(self):
    train = tfx.data.DataSource('train')
    eval = tfx.data.DataSource('eval')
    schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'),
                                    tfx.data.SchemaField.real('y'))
    ds = tfx.data.DataSet.create(schema, train, eval)

    self.assertEqual(ds['train'], train)
    self.assertEqual(ds['eval'], eval)

  def test_empty_dataset_raises_error(self):
    with self.assertRaises(ValueError):
      schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'))
      source = tfx.data.DataSet.create(schema)

  def test_mixed_datasources_raises_error(self):
    class CustomDataSource(tfx.data.DataSource):
      def __init__(self, name):
        super(CustomDataSource, self).__init__(name)
    
    with self.assertRaises(ValueError):
      source1 = tfx.data.DataSource('foo')
      source2 = CustomDataSource('bar')
      schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'))
      ds = tfx.data.DataSet.create(schema, source1, source2)

  def test_parse_local_spec(self):
    spec = {
      'format': 'csv',
      'sources': {
        'train': '/path/to/train.csv',
        'eval': '/path/to/eval.csv'
      }
    }
    schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'))

    ds = tfx.data.DataSet.parse(schema, spec)
    self.assertEqual(len(ds.sources), 2)
    self.assertEqual(ds['train'].path, '/path/to/train.csv')
    self.assertEqual(ds['eval'].path, '/path/to/eval.csv')

  def test_parse_remote_spec(self):
    spec = {
      'format': 'csv',
      'sources': {
        'train': 'https://path/to/train.csv',
        'eval': 'https://path/to/eval.csv'
      }
    }
    schema = tfx.data.Schema.create(tfx.data.SchemaField.integer('x'))

    ds = tfx.data.DataSet.parse(schema, spec)
    self.assertEqual(len(ds), 2)
    self.assertEqual(ds['train'].path, 'https://path/to/train.csv')
    self.assertEqual(ds['eval'].path, 'https://path/to/eval.csv')
