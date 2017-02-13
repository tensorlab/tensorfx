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

# _hooks.py
# Implements various session hooks needed for training.

import logging
import os
import tensorflow as tf
import time
import _utils as utils

from tensorflow.python.lib.io import file_io as tfio


class StopTrainingHook(tf.train.SessionRunHook):
  """Stops training after a specified number of steps.
  """
  def __init__(self, global_steps, max_steps):
    """Initializes an instance of StopTrainingHook.

    Arguments:
      global_steps: The global steps tensor.
      max_steps: The max steps after which training should be stopped.
    """
    self._global_steps = global_steps
    self._max_steps = max_steps

  def before_run(self, context):
    return tf.train.SessionRunArgs(self._global_steps)

  def after_run(self, context, values):
    global_steps_completed = values.results
    if global_steps_completed >= self._max_steps:
      context.request_stop()


class LogSessionHook(tf.train.SessionRunHook):
  """Logs the session loop by outputting steps, and throughput into logs.
  """
  _MESSAGE_FORMAT = 'Run: %.2f sec; Steps: %d; Duration: %d sec; Throughput: %.1f instances/sec'
  def __init__(self, log_steps_interval, batch_size):
    self._log_steps_interval = log_steps_interval
    self._batch_size = batch_size

    self._start_time = time.time()
    self._steps_completed = 0
    self._step_start_time = 0

  def before_run(self, context):
    self._step_start_time = time.time()

  def after_run(self, context, values):
    self._steps_completed += 1

    if self._steps_completed == 1 or \
       self._steps_completed % self._log_steps_interval == 0:
      end_time = time.time()
      run_time = end_time - self._step_start_time
      duration = end_time - self._start_time
      throughput = self._steps_completed * float(self._batch_size) / float(duration)

      logging.info(LogSessionHook._MESSAGE_FORMAT,
                   run_time, self._steps_completed, duration, throughput)


class LogTrainingHook(tf.train.SessionRunHook):
  """Logs the training job by logging progress as well as producing summary events.
  """
  _MESSAGE_FORMAT = 'Global steps: %d; Duration: %d sec; Throughput: %.1f instances/sec; Loss: %.3f'
  def __init__(self, global_steps, loss, summary_op, log_steps_interval, batch_size, output):
    self._global_steps = global_steps
    self._loss = loss
    self._summary_op = summary_op

    self._log_steps_interval = log_steps_interval
    self._batch_size = batch_size

    summary_path = os.path.join(output, 'summaries', 'train')
    self._summary_writer = tf.train.SummaryWriter(summary_path)
    self._summary_writer.add_graph(tf.get_default_graph())

    self._start_time = time.time()
    self._global_steps_completed = 0

  def before_run(self, context):
    current_step = self._global_steps_completed + 1
    if current_step % self._log_steps_interval == 0:
      return tf.train.SessionRunArgs([self._global_steps, self._loss, self._summary_op])
    else:
      return tf.train.SessionRunArgs([self._global_steps])

  def after_run(self, context, values):
    if len(values.results) == 1:
      self._global_steps_completed, = values.results
    else:
      self._global_steps_completed, loss_value, summary = values.results

      end_time = time.time()
      duration = end_time - self._start_time
      throughput = self._global_steps_completed * float(self._batch_size) / float(duration)

      logging.info(LogTrainingHook._MESSAGE_FORMAT,
                   self._global_steps_completed, duration, throughput, loss_value)

      self._summary_writer.add_summary(summary, self._global_steps_completed)
      utils.add_summary_value(self._summary_writer, 'throughput', throughput,
                              self._global_steps_completed)


class SaveCheckpointHook(tf.train.SessionRunHook):
  """Saves checkpoints at regular interval, and then runs evaluation.

  This should only be used in master tasks.
  """
  # TODO: Implement this
  pass


class CheckNaNLossHook(tf.train.SessionRunHook):
  """Checks for NaN loss values to stop or abort training.
  """
  # TODO: Implement this
  pass


