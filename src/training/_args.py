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

# _args.py
# Defines ModelArguments and related classes.

import enum

class JobLogging(enum.Enum):
  """Defines the logging level options for the job.
  """
  FATAL = 50
  ERROR = 40
  WARN = 30
  INFO = 20
  DEBUG = 10


class ModelArguments(object):
  """An object that defines various arguments used to build and train models.
  """
  # Arguments related to training data and reading
  batch_size = 128
  epochs = 0

  # Arguments related to training session loop
  max_steps = 1000

  # Arguments related to diagnostics
  log_level = JobLogging.WARN
  log_device_placement = False

  # Internal
  _args = None
