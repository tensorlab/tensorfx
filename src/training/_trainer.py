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
from ._cluster import Cluster
from ._config import Configuration


class Trainer(object):
  """Provides the functionality to train a model during a training job.
  """
  def __init__(self, model_builder, cluster=None):
    """Initializes an instance of a Trainer.

    Arguments:
      model_builder: The ModelBuilder object that creates graphs representing the model.
      cluster: An optional cluster providing the implementation of training tasks.
    """
    if not cluster:
      # Use default implementation of the Cluster and associated tasks.
      cluster = Cluster()
    
    self._builder = model_builder
    self._cluster = cluster

  def train(self, args, dataset, output=None, config=None):
    """Runs the training process to train a model.

    In the case of master nodes (or single node training), the result is the trained model.

    Arguments:
      args: any arguments, including hyperparameters to parameterize the model to train.
      dataset: the training and evaluation data sources to use during training.
      output: an optional output location to produce checkpoints, summaries, and the model.
      config: an optional configuration providing information about the training job and cluster.
    Returns:
      The result of training. The resulting value is only relevant for master nodes.
    """
    if not output:
      # If an output location is not specified, default to the current working directory
      output = os.getcwd()
    if not config:
      # By default, use the configuration specified in the TF_CONFIG environment variable.
      config = Configuration.environment()

    # TODO: Initialize logging levels, and logging handlers

    # Create the TensorFlow server. This is only needed for distributed training.
    server = None
    if config.distributed:
      server = tf.train.Server(config.cluster, config.task.type, config.task.index,
                               protocol='grpc')

    task = self._cluster.create_task(config)
    return task.run(server, builder=self._builder, args=args, dataset=dataset, output=output)
