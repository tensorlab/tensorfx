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

# _scaffold.py
# Implements ScaffoldCommand


class ScaffoldCommand(object):
  """Implements the tfx scaffold command.
  """
  name = 'scaffold'
  help = 'Createa a new project from a template.'
  extra = False

  @staticmethod
  def build_parser(parser):
    parser.add_argument('--dir', metavar='path', type=str,
                        help='The directory to use to instantiate the template')
    parser.add_argument('--model', metavar='type', type=str, required=True,
                        help='The model to create; eg. "nn.FeedForwardClassification", or "Custom"')

  @staticmethod
  def run(args):
    # TODO: Implement this
    print 'Scaffolding is not yet implemented'
