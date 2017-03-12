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

# _args.py
# Defines ModelArguments and related classes.

import argparse
import logging
import sys
import tensorfx as tfx


class ModelArguments(argparse.Namespace):

  def process(self):
    """Processes the parsed arguments to produce any additional objects.
    """
    # Convert strings to logging values
    self.log_level = getattr(logging, self.log_level)
    self.log_level_tensorflow = getattr(logging, self.log_level_tensorflow)

  @classmethod
  def default(cls):
    """Creates an instance of the arguments with default values.

    Returns:
      The model arguments with default values.
    """
    return cls.parse(args=[])

  @classmethod
  def parse(cls, args=None, parse_job=False):
    """Parses training arguments.

    Arguments:
      args: the arguments to parse. If unspecified, the process arguments are used.
      parse_job: whether to parse the job related standard (input and output) arguments.
    Returns:
      The parsed arguments.
    """
    if args is None:
      args = sys.argv[1:]

    argparser = ModelArgumentsParser(add_job_arguments=parse_job)
    cls.init_parser(argparser)

    args_object = argparser.parse_args(args, namespace=cls())
    args_object._args = args
    args_object.process()

    return args_object

  @classmethod
  def init_parser(cls, parser):
    """Initializes the argument parser.

    Args:
      parser: An argument parser instance to be initialized with arguments.
    """
    session = parser.add_argument_group(title='Session',
                                        description='Arguments controlling the session loop.')
    session.add_argument('--max-steps', type=int, default=1000,
                         help='The number of steps to execute during the training job.')
    session.add_argument('--batch-size', type=int, default=128,
                         help='The number of instances to read and process in each training step.')
    session.add_argument('--epochs', type=int, default=0,
                         help='The number of passes over the training data to make.')
    session.add_argument('--checkpoint-interval-secs', type=int, default=60 * 5,
                         help='The frequency of checkpoints to create during the training job.')

    log_levels = ['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG']

    log = parser.add_argument_group(title='Logging and Diagnostics',
                                    description='Arguments controlling logging during training.')
    log.add_argument('--log-level-tensorflow', metavar='level', type=str, default='ERROR',
                     choices=log_levels,
                     help='The logging level for TensorFlow generated log messages.')
    log.add_argument('--log-device-placement', default=False, action='store_true',
                     help='Whether to log placement of ops and tensors on devices.')
    log.add_argument('--log-level', metavar='level', type=str, default='INFO', choices=log_levels,
                     help='The logging level for training.')
    log.add_argument('--log-interval-steps', metavar='steps', type=int, default=100,
                     help='The frequency of training logs and summary events to generate.')


class ModelArgumentsParser(argparse.ArgumentParser):

  def __init__(self, add_job_arguments):
    # TODO: Add description, epilogue, etc.
    super(ModelArgumentsParser, self).__init__(prog='trainer', usage='%(prog)s [--help] [options]')
    self.var_args_action = AddVarArgAction

    job = self.add_argument_group(title='Job',
                                  description='Arguments defining job inputs and outputs.')
    job.add_argument('--data-schema', metavar='path', type=str, required=False,
                     help='The schema (columns, types) of the data being referenced (YAML).')
    job.add_argument('--data-metadata', metavar='path', type=str, required=False,
                     help='The statistics and vocabularies of the data being referenced (JSON).')
    job.add_argument('--data-features', metavar='path', type=str, required=False,
                     help='The set of features to transform the raw data into (YAML).')
    job.add_argument('--data-train', metavar='path', type=str, required=False,
                     help='The data to use for training. This can include wildcards.')
    job.add_argument('--data-eval', metavar='path', type=str, required=False,
                     help='The data to use for evaluation. This can include wildcards.')

    # The framework uses output, but Cloud ML Engine uses job-dir. Only one should be provided.
    job.add_argument('--output', type=str, dest='output', required=False,
                     help='The output path to use for training outputs,')
    job.add_argument('--job-dir', type=str, dest='output', required=False,
                     help='For Cloud ML Engine compatibility only. Use --output instead.')

  def _parse_optional(self, arg_string):
    suffix_index = arg_string.find(':')
    if suffix_index < 0:
      return super(ModelArgumentsParser, self)._parse_optional(arg_string)

    original_arg_string = arg_string
    suffix = arg_string[suffix_index + 1:]
    arg_string = arg_string[0:suffix_index]

    option_tuple = super(ModelArgumentsParser, self)._parse_optional(arg_string)
    if not option_tuple:
      return option_tuple

    action, option_string, explicit_arg = option_tuple
    if isinstance(action, AddVarArgAction):
      return action, suffix, explicit_arg
    else:
      self.exit(-1, message='Unknown argument %s' % original_arg_string)


class AddVarArgAction(argparse.Action):
  def __init__(self,
               option_strings,
               dest,
               nargs=None,
               const=None,
               default=None,
               type=None,
               choices=None,
               required=False,
               help=None,
               metavar=None):
    super(AddVarArgAction, self).__init__(
      option_strings=option_strings,
      dest=dest,
      nargs=nargs,
      const=const,
      default=default,
      type=type,
      choices=choices,
      required=required,
      help=help,
      metavar=metavar)

  def __call__(self, parser, namespace, values, option_string=None):
    index = 0
    try:
      index = int(option_string) - 1
    except ValueError:
      pass

    list = getattr(namespace, self.dest)
    if list is None:
      list = []
      setattr(namespace, self.dest, list)

    if index >= len(list):
      list.extend([self.default] * (index + 1 - len(list)))
    list[index] = values
