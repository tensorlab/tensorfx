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

import argparse
import autocli
import os
import sys
import tensorflow as tf
from ._config import Configuration

_TASK_PARAM_SERVER = 'ps'
_TASK_WORKER = 'worker'
_TASK_MASTER = 'master'


class Trainer(object):
  """Provides the functionality to train a model during a training job.
  """
  def train(self, model_builder_type, args=None, config=None):
    """Runs the training process to train a model.

    Arguments:
      model_builder_type: The type of ModelBuilder to use to create graphs representing the model.
      args: an optional list of arguments to customize training.
      config: an optional configuration providing information about the training job and cluster.
    Returns:
      The result of training. The resulting value is only relevant for master nodes.
    """
    if not args:
      # By default, just use the process arguments.
      args = sys.argv
    if not config:
      # By default, use the configuration specified in the TF_CONFIG environment variable.
      config = Configuration.environment()

    self._init_tensorflow()

    self._server = None
    if config.distributed:
      self._server = self._create_tensorflow_server(config)
      if config.task.type == _TASK_PARAM_SERVER:
        return self._run_ps()

    self._model_builder = self._create_model_builder(model_builder_type, args, config)
    return self._run_training()

  def _run_ps(self):
    """Runs the parameter server task.

    A ps task runs forever (until killed) using implementation within TensorFlow runtime.
    """
    try:
      self._server.join()
    except AbortError:
      pass

  def _run_training(self):
    """Runs the worker and master tasks.

    Worker and master tasks create a TensorFlow session, and run the session loop. The session
    loop is customized via session hooks. A worker simply runs the training logic, while a master
    is also responsible for producing and evaluating checkpoints, as well producing summary event
    logs, and finally exporting the trained model.
    """
    training = self._model_builder.training()

    with training.graph.as_default() as graph:
      config = self._create_tensorflow_config()
      master = self._server.target if self._server else ''
      scaffold = training.scaffold

      if self._model_builder.config.task.type == _TASK_MASTER:
        checkpoints = os.path.join(self._model_builder.output, 'checkpoints')
        session_creator = tf.train.ChiefSessionCreator(scaffold, master, config, checkpoints)
      else:
        session_creator = tf.train.WorkerSessionCreator(scaffold, master, config)

      with tf.train.MonitoredSession(session_creator, session_hooks) as session:
        while not session.should_stop():
          session.run()

  def _create_model_builder(self, model_builder_type, args, config):
    """Creates the ModelBuilder that is used for training.
    """
    # Parse arguments. Arguments include standard args (inputs and output) which are parsed out first.
    # The remaining arguments are handled as custom args.
    argparser = argparse.ArgumentParser(add_help=False)
    argparser.add_argument('--train', dest='train', type=str, required=True)
    argparser.add_argument('--eval', dest='eval', type=str, required=True)
    argparser.add_argument('--job_dir', dest='output', type=str, default='')
    std_args, custom_args = argparser.parse_known_args(args)

    # TODO: Create the dataset object for input data.
    dataset = None

    # Use 'output' subdirectory of working directory, if the output location has not been specified.
    output = std_args.output if std_args.output else os.path.join(os.getcwd(), 'output')

    # By convention, custom or model specific args are used to initialize an Arguments object that
    # is defined alongside the ModelBuilder in the same module.
    model_builder_module = sys.modules[model_builder_type.__module__]
    model_args_type = model_builder_module.__dict__[model_builder_type.__name__ + 'Arguments']
    model_args = autocli.parse_object(model_args_type, custom_args)

    return _model_builder.create(model_args, dataset, output, config)

  def _create_tensorflow_config(self):
    """Creates the TensorFlow session config object.
    """
    return tf.ConfigProto(log_device_placement=True)

  def _create_tensorflow_server(self, config):
    """Creates the TensorFlow server, which is required for distributed training.
    """
    return tf.train.Server(config.cluster, config.task.type, config.task.index,
                            protocol='grpc')

  def _init_tensorflow(self):
    """Initializes the TensorFlow runtime.

    This initializes global settings, such as logging.
    """
    # TODO: Implement this
    pass
