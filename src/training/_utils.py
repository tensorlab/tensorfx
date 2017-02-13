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

# _utils.py
# Implements various training helpers.

import os
import tensorflow as tf
import tensorflow.core.framework.summary_pb2 as summaries
import yaml

from tensorflow.python.lib.io import file_io as tfio

def save_job_spec(output, config, dataset, args):
  job = {
    'config': config._env,
    'data': dataset._refs,
    'args': ' '.join(args._args)
  }
  job_definition = yaml.safe_dump(job, default_flow_style=False)
  job_file = os.path.join(output, 'job.yaml')

  tfio.recursive_create_dir(output)
  tfio.write_string_to_file(job_file, job_definition)


def add_summary_value(summary_writer, tag, value, global_steps):
  summary_value = summaries.Summary.Value(tag=tag, simple_value=value)
  summary = summaries.Summary(value=[summary_value])

  summary_writer.add_summary(summary, global_steps)
