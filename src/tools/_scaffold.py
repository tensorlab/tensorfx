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

import os
import tensorfx as tfx

class ScaffoldCommand(object):
  """Implements the tfx scaffold command.
  """
  name = 'scaffold'
  help = 'Createa a new project from a template.'
  extra = False

  @staticmethod
  def build_parser(parser):
    parser.add_argument('--name', metavar='name', type=str, required=True,
                        help='The name of the model to use when instantiating the template')
    parser.add_argument('--dir', metavar='path', type=str, required=False, default=os.getcwd(),
                        help='The directory in which to instantiate the template')
    parser.add_argument('--model', metavar='type', type=str, required=False, default='custom',
                        help='The type of model to create; eg. "nn.FeedForwardClassification"')

  @staticmethod
  def run(args):
    variables = {
      'name': args.name,
      'tensorfx_version': tfx.__version__
    }

    contents = {
      'setup.py': _scaffold_setup_py.format(**variables),
      'trainer/__init__.py': _scaffold_trainer_init_py.format(**variables),
    }

    if args.model == 'custom':
      variables['model_class'] = args.name[0].upper() + args.name[1:]
      contents['trainer/main.py'] = _scaffold_trainer_main_py_custom.format(**variables)
      contents['trainer/model.py'] = _scaffold_trainer_model_py.format(**variables)
    else:
      variables['model'] = args.model
      variables['model_set'] = args.model.split('.')[0]
      contents['trainer/main.py'] = _scaffold_trainer_main_py.format(**variables)

    scaffold_path = os.path.join(args.dir, args.name)
    for path, content in contents.iteritems():
      content_path = os.path.join(scaffold_path, path)

      content_dir = os.path.dirname(content_path)
      if not os.path.isdir(content_dir):
        os.makedirs(content_dir)

      with open(content_path, 'w') as content_file:
        content_file.write(content)


# TODO: Externalize these into a template directory

_scaffold_setup_py = """# setup.py

import setuptools

# The name and version of the package.
name = '{name}'
version = '1.0'

# The main modules in the package.
trainer_main = '{name}.trainer.main'


def main():
  \"""Invokes setup to build or install a distribution of the package.
  \"""
  setuptools.setup(name=name, version=version,
                   packages=setuptools.find_packages(),
                   install_requires=[
                     'tensorfx={tensorfx_version}'
                   ])


if __name__ == '__main__':
  main()
"""

_scaffold_trainer_init_py = """# __init__.py
# Declaration of {name}.trainer module.
"""

_scaffold_trainer_main_py = """# main.py
# Implementation of training module.

import tenosrflow as tf
import tensorfx as tfx
import tensorfx.models.{model_set} as {model_set}

args = {model}Arguments.parse(parse_job=True)
dataset = tfx.data.CsvDataSet(args.data_schema,
                              train=args.data_train,
                              eval=args.data_eval,
                              metadata=args.data_metadata,
                              features=args.data_features)

builder = {model}(args)

trainer = tfx.training.ModelTrainer()
model = trainer.train(builder, dataset, args.output)
"""

_scaffold_trainer_main_py_custom = """# main.py
# Implementation of training module.

import tensorflow as tf
import tensorfx as tfx
import _model as model

args = model.{model_class}Arguments.parse(parse_job=True)
dataset = tfx.data.CsvDataSet(args.data_schema,
                              train=args.data_train,
                              eval=args.data_eval,
                              metadata=args.data_metadata,
                              features=args.data_features)

builder = model.{model_class}(args)

trainer = tfx.training.ModelTrainer()
model = trainer.train(builder, dataset, args.outupt)
"""

_scaffold_trainer_model_py = """# model.py
# Implementation of model module.

import tensorflow as tf
import tensorfx as tfx

class {model_class}Arguments(tfx.training.ModelArguments):
  \"""Declares arguments supported by the model.
  \"""
  @classmethod
  def init_parser(cls, parser):
    super({model_class}Arguments, cls).init_parser(parser)

    # TODO: Add additional model-specific arguments.


class {model_class}(tfx.training.ModelBuilder):
  \"""Builds the graphs for training, evaluating and predicting with the model.
  \"""
  def __init__(self, args, dataset):
    super({model_class}, self).__init__(args, dataset)

  # TODO: Implement one or more of the graph building methods. These include one or more of
  # build_input(), build_inference(), build_training(), build_output(), and build_evaluation() or
  # build_training_graph(), build_evaluation_graph(), and build_prediction_graph().
  # See the documentation for more details.
"""
