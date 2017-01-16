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

# _task_worker.py
# Implements WorkerTask.

from ._task import ClusterTask
from ._task import ClusterTaskType


class WorkerTask(ClusterTask):
  """Implements the worker task, which runs the training loop.
  """
  def __init__(self, config, type=ClusterTaskType.Worker):
    """Initializes an instance of a WorkerTask.

    Arguments:
      config: the training configuration.
      type: the type of the task.
    """
    super(ParamServerTask, self).__init__(config, type)

  def run(self, server):
    """Runs the task.

    Arguments:
      server: the TensorFlow server.
    """
    raise NotImplementedError('Implement this')
