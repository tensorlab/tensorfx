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

import tensorflow as tf


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
    steps = values.results
    if steps >= self._max_steps:
      context.request_stop()


class LogTrainingHook(tf.train.SessionRunHook):
  """Logs specific tensors at regular interval into the info log.
  """
  # TODO: Implement this
  pass


class SaveSummaryHook(tf.train.SessionRunHook):
  """Logs summary events at regular interval.

  This should only be used in master tasks.
  """
  # TODO: Implement this
  pass


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


