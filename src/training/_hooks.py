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
import tensorfx as tfx
import time
from tensorflow.core.framework import summary_pb2 as tfsummaries


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
  def __init__(self, args):
    self._log_interval_steps = args.log_interval_steps
    self._batch_size = args.batch_size

    self._start_time = time.time()
    self._steps_completed = 0
    self._step_start_time = 0

  def before_run(self, context):
    self._step_start_time = time.time()

  def after_run(self, context, values):
    self._steps_completed += 1

    if self._steps_completed == 1 or \
       self._steps_completed % self._log_interval_steps == 0:
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
  def __init__(self, args, output, training):
    self._global_steps = training.global_steps
    self._loss = training.loss
    self._summary_op = training.summary_op

    self._log_interval_steps = args.log_interval_steps
    self._max_steps = args.max_steps
    self._batch_size = args.batch_size

    self._summary_writer = tf.summary.FileWriter(os.path.join(output, 'summaries', 'train'))
    self._summary_writer.add_graph(training.graph)

    self._start_time = time.time()
    self._global_steps_completed = 0

  def before_run(self, context):
    current_step = self._global_steps_completed + 1
    if (current_step % self._log_interval_steps == 0) or \
       (current_step + 1 >= self._max_steps):
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
      _log_summary_value(self._summary_writer, 'metrics/throughput', throughput,
                         self._global_steps_completed)


class SaveCheckpointHook(tf.train.SessionRunHook):
  """Saves checkpoints during training, evaluates them, and exports the final checkpoint as a model.

  This should only be used in master tasks.
  """
  _MESSAGE_FORMAT = 'Global steps: %d; Evaluation metric: %.3f'
  def __init__(self, args, output, training, evaluation, prediction):
    self._global_steps = training.global_steps
    self._saver = training.saver

    self._checkpoint_interval_secs = args.checkpoint_interval_secs

    self._checkpoint_path = os.path.join(output, 'checkpoints')
    self._checkpoint_name = os.path.join(self._checkpoint_path, 'model.ckpt')

    self._last_save_time = time.time()
    self._last_save_steps = 0

    self._evaluation = evaluation
    self._summary_writer = tf.summary.FileWriter(os.path.join(output, 'summaries', 'eval'))
    self._summary_writer.add_graph(evaluation.graph)

    self._prediction = prediction
    self._output = output

  def before_run(self, context):
    # Save a checkpoint after the first step (this produces early evaluation results), as well as,
    # every checkpoint interval.
    if self._last_save_steps == 0 or \
       time.time() - self._last_save_time >= self._checkpoint_interval_secs:
      return tf.train.SessionRunArgs([self._global_steps])

  def after_run(self, context, values):
    if values.results:
      global_steps_completed, = values.results
      checkpoint = self._saver.save(context.session, self._checkpoint_name, global_steps_completed)
      self._evaluate(checkpoint, global_steps_completed)

      self._last_save_steps = global_steps_completed
      self._last_save_time = time.time()

  def end(self, session):
    global_steps_completed = session.run(self._global_steps)
    if global_steps_completed != self._last_save_steps:
      checkpoint = self._saver.save(session, self._checkpoint_name, global_steps_completed)
      self._evaluate(checkpoint, global_steps_completed)
      self._export(checkpoint)

  def _evaluate(self, checkpoint, global_steps_completed):
    with self._evaluation.graph.as_default():
      with tf.Session() as session:
        self._evaluation.init_op.run()
        self._evaluation.saver.restore(session, checkpoint)
        self._evaluation.local_init_op.run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
          while not coord.should_stop():
            session.run(self._evaluation.eval_op)
        except tf.errors.OutOfRangeError:
          # Ignore the error raised at the end of an epoch of eval data.
          pass
        finally:
          coord.request_stop()
        coord.join(threads)

        metric_value = session.run(self._evaluation.metric)

        summary = session.run(self._evaluation.summary_op)
        self._summary_writer.add_summary(summary, global_steps_completed)

        logging.info(SaveCheckpointHook._MESSAGE_FORMAT, global_steps_completed, metric_value)

  def _export(self, checkpoint):
    summary_writer = tf.summary.FileWriter(os.path.join(self._output, 'summaries', 'prediction'))
    summary_writer.add_graph(self._prediction.graph)
    summary_writer.close()

    with self._prediction.graph.as_default():
      with tf.Session() as session:
        self._prediction.init_op.run()
        self._prediction.saver.restore(session, checkpoint)
        self._prediction.local_init_op.run()

        model_path = os.path.join(self._output, 'model')
        tfx.prediction.Model.save(session, model_path,
                                  self._prediction.inputs, self._prediction.outputs)


class CheckNaNLossHook(tf.train.SessionRunHook):
  """Checks for NaN loss values to stop or abort training.
  """
  # TODO: Implement this
  pass


def _log_summary_value(summary_writer, tag, value, global_steps):
  summary_value = tfsummaries.Summary.Value(tag=tag, simple_value=value)
  summary = tfsummaries.Summary(value=[summary_value])

  summary_writer.add_summary(summary, global_steps)
