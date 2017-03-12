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

# run.py
# Demonstrates a standalone client that uses in-memory data (using a pandas DataFrame), in-code
# definition of the dataset schema, features and the model, training, and finally predictions.

import json
import pandas as pd
import tensorfx as tfx
import tensorfx.models.nn as nn


def create_dataset():
  """Programmatically build the DataSet
  """
  # Load data into DataFrame objects.
  columns = ['species', 'petal_length', 'petal_width', 'sepal_length', 'sepal_width']
  df_train = pd.read_csv('iris/data/train.csv', names=columns)
  df_eval = pd.read_csv('iris/data/eval.csv', names=columns)

  df_train['species'] = df_train['species'].astype('category')
  df_eval['species'] = df_eval['species'].astype('category')

  # NOTE: Ordinarily, this would be specified in YAML configuration, but defined in code to
  # demonstrate the programmatic interface to FeatureSet and Feature objects. This is equivalent
  # to features.yaml.
  features = [
    tfx.data.Feature.concatenate('X',
      tfx.data.Feature.scale('pl', 'petal_length'),
      tfx.data.Feature.scale('pw', 'petal_width'),
      tfx.data.Feature.scale('sl', 'sepal_length'),
      tfx.data.Feature.scale('sl', 'sepal_width')),
    tfx.data.Feature.target('Y', 'species')
  ]

  return tfx.data.DataFrameDataSet(features=tfx.data.FeatureSet.create(features),
                                   train=df_train, eval=df_eval)


def create_args():
  """Programmatically create the arguments.
  """
  # Build the arguments (programmatically starting with defaults, instead of parsing the
  # program's command-line flags using parse().
  args = nn.FeedForwardClassificationArguments.default()
  args.batch_size = 5
  args.max_steps = 2000
  args.checkpoint_interval_secs = 1
  args.hidden_layers = [('l1', 20, 'relu'), ('l2', 10, 'relu')]

  return args


def main():
  args = create_args()
  dataset = create_dataset()

  # Define the model and the trainer to train the model
  classification = nn.FeedForwardClassification(args)
  trainer = tfx.training.ModelTrainer()

  # Train; since this is training in-process (i.e. by default single node training), the training
  # process is run as the 'master' node, which happens to load and return the exported model that
  # can conveniently be used to produce predictions.
  print 'Training...'
  model = trainer.train(classification, dataset, output='/tmp/tensorfx/iris/df')

  # Predict; predictions are returned as a set of dictionaries, in the same order as the input
  # instances.
  print 'Predicting...'
  instances = [
    '6.3,3.3,6,2.5',   # virginica
    '4.4,3,1.3,0.2',   # setosa
    '6.1,2.8,4.7,1.2'  # versicolor
  ]
  predictions = model.predict(instances)

  # Print out instances and corresponding predictions
  print ''
  for instance, prediction in zip(instances, predictions):
    print '%s -> %s\n' % (instance, json.dumps(prediction, indent=2))


if __name__ == '__main__':
  main()
