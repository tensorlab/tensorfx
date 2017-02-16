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
import logging
import os
import sys
import tensorflow as tf
import tensorfx as tfx
import yaml
from _config import Configuration
from _hooks import *

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
      argparser.add_argument('--job_dir', dest='output', type=str, required=False, default='output')
      job_args, model_args = argparser.parse_known_args(args)

      output = os.path.join(os.getcwd(), job_args.output)

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
    self._setup_logging(model_builder)

    server = None
    if self._config.distributed:
      server = self._create_server()
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
    args = model_builder.args

    self._save_job_spec(dataset, args, output)

    # TODO: Create model_builder.build_interfaces, and add properties for each
    training = model_builder.training(dataset)
    evaluation = model_builder.evaluation(dataset)
    prediction = model_builder.prediction(dataset)

    with training.graph.as_default() as graph:
      master = server.target if server else ''
      config = self._create_session_config(training, args)
      scaffold = self._create_session_scaffold(training, args)
      hooks = self._create_session_hooks(training, evaluation, prediction, args, output)

      if self._config.master:
        checkpoint_path = os.path.join(output, 'checkpoints')
        tfio.recursive_create_dir(checkpoint_path)

        session_creator = tf.train.ChiefSessionCreator(scaffold, master, config, checkpoint_path)
      else:
        session_creator = tf.train.WorkerSessionCreator(scaffold, master, config)

      with tf.train.MonitoredSession(session_creator, hooks) as session:
        while not session.should_stop():
          session.run(training.train_op)

      # if self._config.master:
      #   return tfx.prediction.Model.load(os.path.join(output, 'model'))
      # else:
      #   return None

  def _create_server(self):
    """Creates the TensorFlow server, which is required for distributed training.
    """
    return tf.train.Server(self._config.cluster, self._config.task.type, self._config.task.index,
                            protocol='grpc')

  def _create_session_config(self, training, args):
    """Creates the TensorFlow session config object.
    """
    # TODO: Setup device filters, parallelization, and run timeouts
    return tf.ConfigProto(log_device_placement=args.log_device_placement)

  def _create_session_hooks(self, training, evaluation, prediction, args, output):
    """Creates the TensorFlow session hooks that customize the session loop.
    """
    hooks = []

    hooks.append(LogSessionHook(args))
    if self._config.master:
      hooks.append(LogTrainingHook(args, output, training))
      hooks.append(SaveCheckpointHook(args, output, training, evaluation, prediction))
    hooks.append(StopTrainingHook(training.global_steps, args.max_steps))

    return hooks

  def _create_session_scaffold(self, training, args):
    """Creates a TensorFlow Scaffold that will be associated with the Session.
    """
    scaffold = tf.train.Scaffold(init_op=training.init_op,
                                 local_init_op=training.local_init_op,
                                 ready_op=training.ready_op,
                                 summary_op=training.summary_op,
                                 saver=training.saver)
    scaffold.finalize()

    return scaffold

  def _save_job_spec(self, dataset, args, output):
    job_info = {
      'config': self._config._env,
      'data': dataset._refs,
      'args': ' '.join(args._args)
    }
    job_spec = yaml.safe_dump(job_info, default_flow_style=False)
    job_file = os.path.join(output, 'job.yaml')

    tfio.recursive_create_dir(output)
    tfio.write_string_to_file(job_file, job_spec)

  def _setup_logging(self, model_builder):
    tf.logging.set_verbosity(model_builder.args.log_level.value)

    if hasattr(self._config.job, 'local'):
      # Additional setup to output logs to console for local runs. On cloud, this should
      # be handled by the environment.
      if self._config.distributed:
        format = '%%(levelname)s %s:%d: %%(message)s'
        format = format % (self._config.task.type, self._config.task.index)
      else:
        format = '%(levelname)s: %(message)s'
      
      handler = logging.StreamHandler(stream=sys.stderr)
      handler.setFormatter(logging.Formatter(fmt=format))

      logger = logging.getLogger()
      logger.addHandler(handler)
      logger.setLevel(logging.INFO)
