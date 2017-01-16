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
# Tests dataset related functionality in tensorfx.

import unittest

import tensorfx.data as tfxdata

class TestCases(unittest.TestCase):

  def test_create_dataset(self):
    source = tfxdata.DataSource('foo')
    ds = tfxdata.DataSet.create(source)

    self.assertEqual(ds['foo'], source)

  def test_empty_dataset_raises_error(self):
    with self.assertRaises(ValueError):
      source = tfxdata.DataSet.create()

  def test_mixed_datasources_raises_error(self):
    class CustomDataSource(tfxdata.DataSource):
      def __init__(self, name):
        super(CustomDataSource, self).__init__(name)
    
    with self.assertRaises(ValueError):
      source1 = tfxdata.DataSource('foo')
      source2 = CustomDataSource('bar')
      ds = tfxdata.DataSet.create(source1, source2)
