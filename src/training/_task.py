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

# _task.py
# Implements ClusterTask.

import enum

class ClusterTaskType(enum.Enum):
  """Defines the types of tasks supported in the training cluster.
  """
  Master = 'master'
  Worker = 'worker'
  ParamServer = 'ps'


class ClusterTask(object):
  """Base class for all training tasks.
  """
  def __init__(self, config, type):
    """Initializes a ClusterTask with its type and index.
    
    Arguments:
      config: the training configuration.
      type: the type of task represented by this object.
    """
    self._config = config
    self._type = type

  @property
  def config(self):
    """Retrieves the associated TrainingConfig object.
    """
    return self._config

  @property
  def type(self):
    """Retrieves the associated ClusterTaskType value.
    """
    return self._type

  def run(self, server, **kwargs):
    """Runs the task during training.

    Arguments:
      server: The TensorFlow server associated with the current task.
      kwargs: Additional arguments, specific to the type of the task.
    Returns:
      The result of training, based on the type of the task.
    """
    raise NotImplementedError('run must be implemented.')
