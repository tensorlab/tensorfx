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

# _cluster.py
# Implements Cluster.

from ._task import ClusterTaskType
from ._task_ps import ParamServerTask
from ._task_worker import WorkerTask
from ._task_master import MasterTask


class Cluster(object):
  """Provides a standard implementation of a TensorFlow cluster used for training.

  A Cluster is responsible for creating task instances to run during training. A derived class
  can be used to provide alternate implementations to customize training behavior, if needed.
  """
  def __init__(self):
    """Initializes an instance of a Cluster.
    """
    pass

  def create_task(self, config):
    """Creates the task based on the task information in the specified configuration.

    Arguments:
      config: An instance of a Configuration object.
    Returns:
      An instance of a ClusterTask implementation matching the specified configuration.
    Raises:
      ValueError if the specified task type is unknown.
    """
    try:
      task_type = ClusterTaskType[config.task.type]
      if task_type == ClusterTaskType.Master:
        return self.create_master(config)
      elif task_type == ClusterTaskType.Worker:
        return self.create_worker(config)
      elif task_type == ClusterTaskType.ParamServer:
        return self.create_param_server(config)
    except KeyError:
      pass

    raise ValueError('Unknown task type "%s"' % config.task.type)

  def create_master(self, config):
    """Creates a ClusterTask for the master node.

    Arguments:
      config: An instance of a Configuration object.
    Returns:
      An instance of a ClusterTask implementation matching the specified configuration.
    """
    return MasterTask(config)

  def create_worker(self, config):
    """Creates a ClusterTask for a worker node.

    Arguments:
      config: An instance of a Configuration object.
    Returns:
      An instance of a ClusterTask implementation matching the specified configuration.
    """
    return WorkerTask(config)

  def create_param_server(self, config):
    """Creates a ClusterTask for a parameter server node.

    Arguments:
      config: An instance of a Configuration object.
    Returns:
      An instance of a ClusterTask implementation matching the specified configuration.
    """
    return ParamServerTask(config)
