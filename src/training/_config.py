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

# _config.py
# Implements TrainingConfig.

import json
import os
import tensorflow as tf

_TASK_PARAM_SERVER = 'ps'
_TASK_WORKER = 'worker'
_TASK_MASTER = 'master'


class Configuration(object):
  """Contains configuration information for the training process.
  """
  def __init__(self, task, cluster, job, env):
    """Initializes a TrainingConfig instance from the individual configuration objects.

    Task configuration represents the current training task (for both single node and distributed
    training), while cluster configuration represents the cluster and should be None in single
    node training.
    Job configuration represents any environment-specific representation of the training job,

    Arguments:
      task: current TensorFlow task configuration.
      cluster: containing TensorFlow cluster configuration for distributed training.
      job: environment-specific job configuration.
      env: the environment-provided configuration information.
    """
    self._task = type('TaskSpec', (object,), task)
    self._cluster = tf.train.ClusterSpec(cluster) if cluster else None
    self._job = type('JobSpec', (object,), job)
    self._env = env

  @classmethod
  def environment(cls):
    """Creates a Configuration object for single node and distributed training.

    This relies on looking up configuration from an environment variable, 'TF_CONFIG' which allows
    a hosting environment to configure the training process.
    The specific environment variable is expected to be a JSON formatted dictionary containing
    configuration about the current task, cluster and job.

    Returns:
      A Configuration instance matching the current environment.
    """
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # Note that the lookup for 'task' must handle the case where it is missing, as well as when it
    # is specified, but is empty, to support both single node and distributed training.

    return cls(env.get('task', None) or {'type': 'master', 'index': 0},
               env.get('cluster', None),
               env.get('job', {}),
               env)
  
  @classmethod
  def local(cls):
    """Creates a Configuration object representing single node training in a process.

    Returns:
      A default Configuration instance with simple configuration.
    """
    return cls(task={'type': 'master', 'index': 0}, cluster=None, job={}, env={})

  @property
  def distributed(self):
    """Determines if training being performed is distributed or is single node training.

    Returns:
      True if the configuration represents distributed training; False otherwise.
    """
    return self._cluster is not None

  @property
  def cluster(self):
    """Retrieves the cluster definition containing the current node.

    This is None if the current node is part of a single node training job.
    """
    return self._cluster

  @property
  def job(self):
    """Retrieves the job definition of the current training job.
    """
    return self._job

  @property
  def task(self):
    """Retrieves the task definition associated with the current node.

    If no job information is provided, this is None.
    """
    return self._task

  @property
  def device(self):
    """Retrieve the device associated with the current node.
    """
    return '/job:%s/task:%d' % (self._task.type, self._task.index)

  @property
  def master(self):
    """Retrieves whether the current task is a master task.
    """
    return self._task.type == _TASK_MASTER

  @property
  def param_server(self):
    """Retrieves whether the current task is a parameter server task.
    """
    return self._task.type == _TASK_PARAM_SERVER

  @property
  def worker(self):
    """Retrieves whether the current task is a worker task.
    """
    return self._task.type == _TASK_WORKER

  def create_device_setter(self, args):
    """Creates the device setter, which assigns variables and ops to devices in distributed mode.

    Arguments:
      args: the arguments associated with the current job.
    """
    # TODO: Provide a way to provide a custom stragery or setter
    return tf.train.replica_device_setter(cluster=self._cluster,
                                          ps_device='/job:ps',
                                          worker_device=self.device)

  def create_server(self):
    """Creates the TensorFlow server, which is required for distributed training.
    """
    if not self.distributed:
      return None
    return tf.train.Server(self._cluster, self._task.type, self._task.index, protocol='grpc')
