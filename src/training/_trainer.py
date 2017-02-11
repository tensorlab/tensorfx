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
import tensorfx as tfx
from ._config import Configuration

from tensorflow.python.lib.io import file_io as tfio


class ModelTrainer(object):
  """Provides the functionality to train a model during a training job.
  """
  def __init__(self, config=None):
    """Initializes a ModelTrainer instance.

    Arguments:
      config: an optional configuration providing information about the training job and cluster.
    """
    if not config:
      # By default, use the configuration specified in the TF_CONFIG environment variable.
      config = Configuration.environment()

    self._config = config

  @property
  def config(self):
    """Retrieves the training configuration.
    """
    return self._config

  def parse_args(self, model_args_type, args=None, parse_io_flags=True):
    """Parses arguments into model arguments and optionally input/output arguments.

    The arguments type is inspected for its fields to build a model arguments parser. The result is
    an instance of that type. The autocli library is used for parsing the arguments.

    When parse_io_flags is True, the following standard arguments are parsed into a DataSet and
    output location:
    - train: the training data
    - eval: the eval data
    - schema: the schema of the data
    - metadata: metadata about the fields in the schema inferred from analyzing the training data.
    - features: the features to produce from transforming the data.
    - job_dir: the output location associated with the job.

    Arguments:
      model_args_type: the type of the Arguments object to parse.
      args: the list of arguments to parse (by default this uses the process arguments).
      parse_io_flags: whether to parse input dataset and output location arguments.
    Returns:
      A dataset, output and model args tuple if parse_io_flags is True or just the latter if False.
    """
    if args is None:
      args = sys.argv[1:]

    if parse_io_flags:
      # Arguments include standard args (inputs and output) which are parsed out first.
      # The remaining arguments are handled as model-specific args.
      argparser = argparse.ArgumentParser(add_help=False)
      argparser.add_argument('--train', type=str, required=True)
      argparser.add_argument('--eval', type=str, required=True)
      argparser.add_argument('--schema', type=str, required=True)
      argparser.add_argument('--features', type=str, required=False, default=None)
      argparser.add_argument('--metadata', type=str, required=False, default=None)
      argparser.add_argument('--job_dir', dest='output', type=str, required=False,
                            default=os.path.join(os.getcwd(), 'output'))
      io_args, model_args_list = argparser.parse_known_args(args)

      datasources = {
        'train': io_args.train,
        'eval': io_args.eval
      }
      schema_spec = tfio.read_file_to_string(io_args.schema)
      features = tfio.read_file_to_string(io_args.features) if io_args.features else None
      metadata = tfio.read_file_to_string(io_args.metadata) if io_args.metadata else None

      dataset = tfx.Data.DataSet.parse(schema_spec, datasources,
                                      metadata=metadata,
                                      features=features)

      model_args = autocli.parse_object(model_args_type, model_args_list)
      return dataset, io_args.output, model_args
    else:
      return autocli.parse_object(model_args_type, args)

  def train(self, model_builder, dataset, output):
    """Runs the training process to train a model.

    Arguments:
      model_builder: the ModelBuilder to use to build graphs during training.
      dataset: the DataSet to use for training and evaluation.
      output: the location of the output produced during training.
    Returns:
      The trained Model. The resulting value is only relevant for master nodes.
    """
    self._init_tensorflow()

    server = None
    if self._config.distributed:
      server = self._create_tensorflow_server()
      if config.param_server:
        return self._run_ps(server)

    return self._run_training(server, model_builder, dataset, output)

  def _run_ps(self, server):
    """Runs the parameter server task.

    A ps task runs forever (until killed) using implementation within TensorFlow runtime.
    """
    try:
      server.join()
    except AbortError:
      pass

  def _run_training(self, server, model_builder, dataset, output):
    """Runs the worker and master tasks.

    Worker and master tasks create a TensorFlow session, and run the session loop. The session
    loop is customized via session hooks. A worker simply runs the training logic, while a master
    is also responsible for producing and evaluating checkpoints, as well producing summary event
    logs, and finally exporting the trained model.
    """
    training = model_builder.training()

    with training.graph.as_default() as graph:
      master = server.target if server else ''
      config = self._create_tensorflow_session_config()
      scaffold = self._create_tensorflow_session_scaffold(training)
      hooks = self._create_tensorflow_session_hooks(training)

      if self._config.master:
        checkpoints = os.path.join(output, 'checkpoints')
        session_creator = tf.train.ChiefSessionCreator(scaffold, master, config, checkpoints)
      else:
        session_creator = tf.train.WorkerSessionCreator(scaffold, master, config)

      with tf.train.MonitoredSession(session_creator, hooks) as session:
        while not session.should_stop():
          session.run(training.train_op)

      if self._config.master:
        # TODO: Build a Model and return it
        pass

      return None

  def _create_tensorflow_server(self):
    """Creates the TensorFlow server, which is required for distributed training.
    """
    return tf.train.Server(self._config.cluster, self._config.task.type, self._config.task.index,
                            protocol='grpc')

  def _create_tensorflow_session_config(self):
    """Creates the TensorFlow session config object.
    """
    # TODO: Setup device filters, parallelization, and run timeouts
    return tf.ConfigProto(log_device_placement=True)

  def _create_tensorflow_session_hooks(self, training):
    """Creates the TensorFlow session hooks that customize the session loop.
    """
    # TODO: Implement this
    return []

  def _create_tensorflow_session_scaffold(self, training):
    """Creates a TensorFlow Scaffold that will be associated with the Session.
    """
    # TODO: Implement this
    return tf.train.Scaffold()

  def _init_tensorflow(self):
    """Initializes the TensorFlow runtime.

    This initializes global settings, such as logging.
    """
    # TODO: Implement this
    pass
