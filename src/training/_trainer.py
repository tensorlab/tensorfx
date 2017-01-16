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

from ._cluster import Cluster
from ._config import Configuration


class Trainer(object):
  """Provides the functionality to train a model during a training job.
  """
  def __init__(self, graph_builder, cluster=None):
    """Initializes an instance of a Trainer.

    Arguments:
      graph_builder: The GraphBuilder object that creates TensorFlow graphs representing the model.
      cluster: An optional cluster providing the implementation of training tasks.
    """
    if not cluster:
      # Use default implementation of the Cluster and associated tasks.
      cluster = Cluster()
    
    self._graph_builder = graph_builder
    self._cluster = cluster

  def train(self, dataset, output, args, config=None):
    """Runs the training process to train a model.

    In the case of master nodes (or single node training), the result is the trained model.

    Arguments:
      dataset: the training and evaluation data sources to use during training.
      output: the output location to produce checkpoints, summaries, and the resulting model.
      args: any arguments, including hyperparameters to parameterize the model to train.
      config: an optional configuration providing information about the training job and cluster.
    Returns:
      The result of training. The resulting value is only relevant for master nodes.
    """
    if not config:
      # By default, use the configuration specified in the TF_CONFIG environment variable.
      config = Configuration.environment()

    server = self._create_server(config)
    task = self._cluster.create_task(config)

    # TODO: Correctly implement passing on the graph builder, and other args to worker/master tasks.

    return task.run(server)
    
  def _create_server(self, config):
    """Creates an instance of a TensorFlow server.
    """
    if config.distributed:
      raise NotImplementedError('Implement this')

    # For single node training, i.e. local only training, a TensorFlow server is not required.
    raise None
