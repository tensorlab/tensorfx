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

# _task_ps.py
# Implements ParamServerTask.

from ._task import ClusterTask
from ._task import ClusterTaskType


class ParamServerTask(ClusterTask):
  """Implements the parameter server task, which manages variables during distributed training.

  This class relies on built-in TensorFlow implementation.
  """
  def __init__(self, config):
    """Initializes an instance of a ParamServerTask.

    Arguments:
      config: the training configuration.
    """
    super(ParamServerTask, self).__init__(config, ClusterTaskType.ParamServer)

  def run(self, server):
    """Runs the task.

    A parameter server task runs forever (until killed) using implementation within the
    TensorFlow runtime.

    Arguments:
      server: the TensorFlow server.
    """
    # TODO: Figure out what is the error handling strategy to implement here.
    server.join()
