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
import sys
import tensorflow as tf
import tensorfx as tfx
from _config import Configuration
from _hooks import *
from _job import Job

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

  def parse_args(self, model_args_type, args=None, parse_job_args=True):
    """Parses arguments into model arguments and optionally input/output arguments.

    The arguments type is inspected for its fields to build a model arguments parser. The result is
    an instance of that type. The autocli library is used for parsing the arguments.

    When parse_job_args is True, the following job arguments are parsed into a DataSet and
    output location:
    - train: the training data
    - eval: the eval data
    - format: the format of the data
    - schema: the schema of the data
    - metadata: metadata about the fields in the schema inferred from analyzing the training data.
    - features: the features to produce from transforming the data.
    - job_dir: the output location associated with the job.

    Arguments:
      model_args_type: the type of the Arguments object to parse.
      args: the list of arguments to parse (by default this uses the process arguments).
      parse_io_flags: whether to parse input dataset and output location arguments.
    Returns:
      A dataset, output and model args tuple if parse_job_args is True or just the latter if False.
    """
    if args is None:
      args = sys.argv[1:]

    if parse_job_args:
      # Arguments include standard args (inputs and output) which are parsed out first.
      # The remaining arguments are handled as model-specific args.
      argparser = argparse.ArgumentParser(add_help=False)
      argparser.add_argument('--train', type=str, required=True)
      argparser.add_argument('--eval', type=str, required=True)
      argparser.add_argument('--format', type=str, required=True)
      argparser.add_argument('--schema', type=str, required=True)
      argparser.add_argument('--features', type=str, required=False, default=None)
      argparser.add_argument('--metadata', type=str, required=False, default=None)
      argparser.add_argument('--job_dir', dest='output', type=str, required=True)
      job_args, model_args = argparser.parse_known_args(args)

      schema_spec = tfio.read_file_to_string(job_args.schema)
      features = tfio.read_file_to_string(job_args.features) if job_args.features else None
      metadata = tfio.read_file_to_string(job_args.metadata) if job_args.metadata else None

      dataset_spec = {
        'format': job_args.format,
        'sources': {
          'train': job_args.train,
          'eval': job_args.eval
        }
      }

      output = job_args.output
      references = vars(job_args)
      references.pop('output')

      dataset = tfx.data.DataSet.parse(schema_spec, dataset_spec,
                                       metadata=metadata,
                                       features=features,
                                       refs=references)

      model_args = autocli.parse_object(model_args_type, model_args)
      return dataset, output, model_args
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
    job = Job(output, model_builder, dataset, self._config)
    job.configure_logging()

    server = self._config.create_server()
    if server and self._config.param_server:
      return self._run_ps(server)

    return self._run_training(server, job)

  def _run_ps(self, server):
    """Runs the parameter server task.

    A ps task runs forever (until killed) using implementation within TensorFlow runtime.
    """
    try:
      server.join()
    except AbortError:
      pass

  def _run_training(self, server, job):
    """Runs the worker and master tasks.

    Worker and master tasks create a TensorFlow session, and run the session loop. The session
    loop is customized via session hooks. A worker simply runs the training logic, while a master
    is also responsible for producing and evaluating checkpoints, as well producing summary event
    logs, and finally exporting the trained model.
    """
    job.start()

    with job.training.graph.as_default() as graph:
      master = server.target if server else ''
      config = self._create_session_config(job)
      hooks = self._create_session_hooks(job)

      if self._config.master:
        tfio.recursive_create_dir(job.checkpoints_path)
        session_creator = tf.train.ChiefSessionCreator(job.training.scaffold,
                                                       master, config, job.checkpoints_path)
      else:
        session_creator = tf.train.WorkerSessionCreator(job.training.scaffold, master, config)

      with tf.train.MonitoredSession(session_creator, hooks) as session:
        while not session.should_stop():
          # TODO: Add session run timeouts
          session.run(job.training.train_op)

      if self._config.master:
        return tfx.prediction.Model.load(job.model_path)
      else:
        return None

  def _create_session_config(self, job):
    """Creates the TensorFlow session config object.
    """
    if self._config.local:
      # Don't have each process (esp. in case of distributed simulation) on the local machine to
      # attempt using all CPUs
      parallelism = 1
    else:
      # Use default
      parallelism = 0

    # Limit communication to specific devices. Specifically the goal is to disable communications
    # across workers, so as to increase performance and reliability.
    device_filters = ['/job:ps', self._config.device]

    return tf.ConfigProto(log_device_placement=job.args.log_device_placement,
                          device_filters=device_filters,
                          intra_op_parallelism_threads=parallelism,
                          inter_op_parallelism_threads=parallelism)

  def _create_session_hooks(self, job):
    """Creates the TensorFlow session hooks that customize the session loop.
    """
    hooks = []

    hooks.append(LogSessionHook(job))
    if self._config.master:
      hooks.append(LogTrainingHook(job))
      hooks.append(SaveCheckpointHook(job))
    hooks.append(StopTrainingHook(job))

    return hooks
