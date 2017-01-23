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

# config_tests.py
# Tests config related functionality in tensorfx.

import json
import os
import unittest

import tensorfx as tfx

class TestCases(unittest.TestCase):

  def test_local_config(self):
    config = tfx.training.Configuration.local()

    self.assertFalse(config.distributed)
    self.assertIsNone(config.cluster)
    self.assertIsNotNone(config.task)
    self.assertEqual(config.task.type, 'master')

  def test_empty_env_config(self):
    config = tfx.training.Configuration.environment()

    self.assertFalse(config.distributed)
    self.assertIsNone(config.cluster)
    self.assertIsNotNone(config.task)
    self.assertEqual(config.task.type, 'master')

  def test_env_config(self):
    config = {
      'task': {
        'type': 'master',
        'index': 0
      },
      'cluster': {
        'hosts': []
      }
    }
    os.environ['TF_CONFIG'] = json.dumps(config)

    config = tfx.training.Configuration.environment()

    self.assertTrue(config.distributed)
    self.assertIsNotNone(config.cluster)
    self.assertIsNotNone(config.task)
    self.assertEqual(config.task.type, 'master')

