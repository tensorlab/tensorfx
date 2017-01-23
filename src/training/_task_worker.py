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
    super(WorkerTask, self).__init__(config, type)
    self.args = None
    self.model_builder = None
    self.dataset = None
    self.output = None

  def run(self, server, **kwargs):
    """Runs the task.

    Arguments:
      server: the TensorFlow server.
    """
    self.args = kwargs.get('args')
    self.dataset = kwargs.get('dataset')
    self.output = kwargs.get('output')
    self.builder = kwargs.get('builder')

    self.training = self.builder.training(self.args, self.dataset)
    with self.training.graph.as_default():
      with self.create_session(server) as session:
        pass

  def create_session(self, server):
    """Creates the TensorFlow session within the training task.
    """
    raise NotImplementedError('Implement this')
