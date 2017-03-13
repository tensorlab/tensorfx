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

# train.py
# tensorfx.tools.tfx module to implement the tfx command-line tool.

import argparse
import sys
from _scaffold import ScaffoldCommand
from _train import TrainCommand
from _predict import PredictCommand


def _build_cli():
  """Builds the command-line interface.
  """
  commands = [
    ScaffoldCommand,
    TrainCommand,
    PredictCommand
  ]

  cli = argparse.ArgumentParser(prog='tfx')
  subparsers = cli.add_subparsers(title='Available commands')

  for command in commands:
    command_parser = subparsers.add_parser(command.name, help=command.help,
                                           usage='%(prog)s [--help] [options]')
    command_parser.set_defaults(command=command)
    command.build_parser(command_parser)

  return cli


def main(args=None):
  if not args:
    args = sys.argv[1:]

  cli = _build_cli()
  args, extra_args = cli.parse_known_args(args)

  command = args.command
  del args.command

  if extra_args:
    if command.extra:
      args.extra = extra_args
    else:
      cli.error('unrecognized arguments %s' % ' '.join(extra_args))

  command.run(args)


if __name__ == '__main__':
  main(sys.argv[1:])
