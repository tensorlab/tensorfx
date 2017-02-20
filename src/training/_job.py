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

# _job.py
# Implements Job.

import os
import logging
import yaml
import sys
import tensorflow as tf
from tensorflow.python.lib.io import file_io as tfio

class Job(object):
  """Represents a training job.
  """
  def __init__(self, model_builder, output, config):
    """Initializes a Job instance.

    Arguments:
      model_builder: the ModelBuilder associated with the job.
      output: the output path of the job.
      config: the Training configuration.
    """
    self._model_builder = model_builder
    self._output = output
    self._config = config

  @property
  def args(self):
    """Retrieves the arguments associated with the job.
    """
    return self._model_builder.args

  @property
  def model_builder(self):
    """Retrieves the ModelBuilder being used to build model graphs.
    """
    return self._model_builder

  @property
  def output_path(self):
    """Retrieves the output path of the job.
    """
    return self._output

  @property
  def checkpoints_path(self):
    """Retrieves the checkpoints path within the output path.
    """
    return os.path.join(self._output, 'checkpoints')

  @property
  def model_path(self):
    """Retrieves the model path within the output path.
    """
    return os.path.join(self._output, 'model')

  @property
  def training(self):
    """Retrieves the training graph interface for the job.
    """
    return self._training

  @property
  def evaluation(self):
    """Retrieves the evaluation graph interface for the job.
    """
    return self._evaluation

  @property
  def prediction(self):
    """Retrieves the prediction graph interface for the job.
    """
    return self._prediction

  def summaries_path(self, summary):
    """Retrieves the summaries path within the output path.

    Arguments:
      summary: the type of summary.
    """
    return os.path.join(self._output, 'summaries', summary)

  def configure_logging(self):
    """Initializes the loggers for the job.
    """
    args = self._model_builder.args

    tf.logging.set_verbosity(getattr(tf.logging, args.log_level_tensorflow.name))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.log_level_tensorflow.value)

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
      logger.setLevel(getattr(logging, args.log_level.name))

  def start(self):
    """Performs startup logic, including building graphs.
    """
    if self._config.master:
      # Save out job information for later reference alongside all other outputs.
      job_args = ' '.join(self._model_builder.args._args).replace(' --', '\n--').split('\n')
      job_info = {
        'config': self._config._env,
        'args': job_args
      }
      job_spec = yaml.safe_dump(job_info, default_flow_style=False)
      job_file = os.path.join(self._output, 'job.yaml')

      tfio.recursive_create_dir(self._output)
      tfio.write_string_to_file(job_file, job_spec)

      # Create a checkpoints directory. This is needed to ensure checkpoint restoration logic
      # can lookup an existing directory.
      tfio.recursive_create_dir(self.checkpoints_path)

    # Build the graphs that will be used during the course of the job.
    self._training, self._evaluation, self._prediction = \
      self._model_builder.build_graph_interfaces(self._config)
