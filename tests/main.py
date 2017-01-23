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

# main.py
# Entrypoint for tests

import os
import sys
import unittest

# Add the library being tested to be on the path and then import it
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

# Load the test modules
import data.dataset_tests
import training.config_tests

_TEST_MODULES = [
  data.dataset_tests,
  training.config_tests
]


def main():
  suite = unittest.TestSuite()
  for m in _TEST_MODULES:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))

  runner = unittest.TextTestRunner()
  result = runner.run(suite)

  sys.exit(len(result.errors) + len(result.failures))


if __name__ == '__main__':
  main()

