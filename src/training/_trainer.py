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

# _trainer.py
# Implements Trainer.

import os
import tensorflow as tf
from ._config import Configuration
from ._runner import GraphRunner

_TASK_PARAM_SERVER = 'ps'
_TASK_WORKER = 'worker'
_TASK_MASTER = 'master'


class Trainer(object):
  """Provides the functionality to train a model during a training job.
  """
  def __init__(self, graph_builder, graph_runner=None, config=None):
    """Initializes an instance of a Trainer.

    Arguments:
      graph_builder: The GraphBuilder object that creates graphs representing the model.
      graph_runner: The GraphRunner object that creates a session to run the graph.
      config: an optional configuration providing information about the training job and cluster.
    """
    if not graph_runner:
      # Use default implementation of the GraphRunner.
      graph_runner = GraphRunner()
    if not config:
      # By default, use the configuration specified in the TF_CONFIG environment variable.
      config = Configuration.environment()
    
    self._graph_builder = graph_builder
    self._graph_runner = graph_runner
    self._config = config

  def train(self, args, dataset, output=None):
    """Runs the training process to train a model.

    In the case of master nodes (or single node training), the result is the trained model.

    Arguments:
      args: any arguments, including hyperparameters to parameterize the model to train.
      dataset: the training and evaluation data sources to use during training.
      output: an optional output location to produce checkpoints, summaries, and the model.
    Returns:
      The result of training. The resulting value is only relevant for master nodes.
    """
    if not output:
      # If an output location is not specified, default to the current working directory
      output = os.getcwd()

    self._init_tensorflow()

    server = self._create_server()
    if server and self._config.task.type == _TASK_PARAM_SERVER:
      return self._run_ps(server)

    return self._run_training(server, args, dataset, output)

  def _create_server(self):
    """Creates the TensorFlow server, which is required for distributed training.
    """
    if self._config.distributed:
      return tf.train.Server(self._config.cluster, self._config.task.type, self._config.task.index,
                             protocol='grpc')
    else:
      # A TensorFlow server is not required for non-distributed single process training
      return None

  def _init_tensorflow(self):
    """Initializes the TensorFlow runtime.

    This initializes global settings, such as logging.
    """
    pass

  def _run_ps(self, server):
    """Runs the parameter server task.

    A ps task runs forever (until killed) using implementation within TensorFlow runtime.

    Arguments:
      server: the TensorFlow server.
    """
    try:
      server.join()
    except AbortError:
      pass

  def _run_training(self, server, args, dataset, output):
    """Runs the worker and master tasks.

    Worker and master tasks create a TensorFlow session, and run the session loop. The session
    loop is customized via session hooks. A worker simply runs the training logic, while a master
    is also responsible for producing and evaluating checkpoints, as well producing summary event
    logs, and finally exporting the trained model.

    Arguments:
      server: the TensorFlow server.
      args: any arguments, including hyperparameters to parameterize the model to train.
      dataset: the training and evaluation data sources to use during training.
      output: an optional output location to produce checkpoints, summaries, and the model.
    """
    pass
