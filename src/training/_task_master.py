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

# _task_master.py
# Implements MasterTask.

from ._task import ClusterTaskType
from ._task_worker import WorkerTask


class MasterTask(WorkerTask):
  """Implements the worker task, which runs the training loop, checkpointing, logging and exporting.
  """
  def __init__(self, config):
    """Initializes an instance of a MasterTask.

    Arguments:
      config: the training configuration.
    """
    super(MasterTask, self).__init__(config, type=ClusterTaskType.Master)
