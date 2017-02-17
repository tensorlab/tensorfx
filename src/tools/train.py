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
# tensorfx.tools.train module to provide a local training single-node and distributed launcher.

import argparse
import json
import os
import subprocess
import sys

_PORT = 14000

def main(args, trainer_args):
  cmd = ['python', '-m', args.module] + trainer_args

  if args.distributed:
    print 'Launching training tasks (master, worker, parameter server)...'
    print ' '.join(cmd)
    print '----\n'

    ps_task = _start_task(cmd, _create_distributed_config('ps'))
    master_task = _start_task(cmd, _create_distributed_config('master'))
    worker_task = _start_task(cmd, _create_distributed_config('worker'))
  else:
    print 'Launching training task...'
    print ' '.join(cmd)
    print '----\n'

    master_task = _start_task(cmd, _create_simple_config())
    ps_task = None
    worker_task = None

  try:
    master_task.wait()
  finally:
    if worker_task:
      _kill_task(worker_task)
    if ps_task:
      _kill_task(ps_task)
    _kill_task(master_task)

def _create_simple_config():
  return {
    'task': {'type': 'master', 'index': 0},
    'job': {'local': True}
  }

def _create_distributed_config(task):
  return {
    'cluster': {
      'ps': ['localhost:%d' % _PORT],
      'master': ['localhost:%d' % (_PORT + 1)],
      'worker': ['localhost:%d' % (_PORT + 2)]
    },
    'task': {'type': task, 'index': 0},
    'job': {'local': True}
  }

def _start_task(cmd, config):
  env = os.environ.copy()
  env['TF_CONFIG'] = json.dumps(config)
  return subprocess.Popen(cmd, env=env)

def _kill_task(process):
  try:
    process.terminate()
  except:
    pass

def _parse_args(argv):
  argparser = argparse.ArgumentParser(description='Launches training jobs for development/testing.')
  argparser.add_argument('--module', metavar='name', type=str, required=True,
                         help='The name of the training module to launch')
  argparser.add_argument('--output', metavar='path', type=str, default='output',
                         help='The path to write outputs')
  argparser.add_argument('--distributed', action='store_true',
                         help='Runs a multi-node (master, worker, parameter server) cluster')

  args, trainer_args = argparser.parse_known_args(argv)

  trainer_args.extend([
    '--job_dir', os.path.abspath(args.output)
  ])

  return args, trainer_args


if __name__ == '__main__':
  args, trainer_args = _parse_args(sys.argv[1:])
  main(args, trainer_args)
