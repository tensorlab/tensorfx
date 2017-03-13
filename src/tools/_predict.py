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

# _predict.py
# Implements PredictCommand.

import json
import os
import sys
import tensorflow as tf
import tensorfx as tfx


class PredictCommand(object):
  """Implements the tfx predict command to use a model to produce predictions.
  """
  name = 'predict'
  help = 'Produces predictions using a model.'
  extra = False

  @staticmethod
  def build_parser(parser):
    parser.add_argument('--model', metavar='path', type=str, required=True,
                        help='The path to a previously trained model.')
    parser.add_argument('--input', metavar='path', type=str,
                        help='The path to a file with input instances. Uses stdin by default.')
    parser.add_argument('--output', metavar='path', type=str,
                        help='The path to a file to write outputs to. Uses stdout by default.')
    parser.add_argument('--batch-size', metavar='instances', type=int, default=10,
                        help='The number of instances to predict per batch.')

  @staticmethod
  def run(args):
    # TODO: Figure out where to do JSON and TF initialization in more common way.
    json.encoder.FLOAT_REPR = lambda f: ('%.5f' % f)

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.ERROR)

    model = tfx.prediction.Model.load(args.model)

    with TextSource(args.input, args.batch_size) as source, TextSink(args.output) as sink:
      for instances in source:
        predictions = model.predict(instances)
        lines = map(lambda p: json.dumps(p, sort_keys=True), predictions)
        sink.write(lines)


class TextSource(object):

  def __init__(self, file=None, batch_size=1):
    self._file = file
    self._batch_size = batch_size

  def __enter__(self):
    self._stream = open(self._file, 'r') if self._file else sys.stdin
    return self

  def __exit__(self, type, value, traceback):
    if self._stream and self._file:
      self._stream.close()

  def __iter__(self):
    instances = []

    while True:
      instance = self._stream.readline().strip()
      if not instance:
        # EOF
        break

      instances.append(instance)
      if len(instances) == self._batch_size:
        # A desired batch of instances is available
        yield instances
        instances = []

    if instances:
      yield instances


class TextSink(object):

  def __init__(self, file=None):
    self._file = file

  def __enter__(self):
    self._stream = open(self._file, 'w') if self._file else sys.stdout
    return self

  def __exit__(self, type, value, traceback):
    if self._stream and self._file:
      self._stream.close()

  def write(self, lines):
    for l in lines:
      self._stream.write(l + '\n')
