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

# data.py
# A utility to generate data in TF.Example protobufs saved into a TF.Record file.

import pandas as pd
import tensorflow as tf
import tensorflow.core.example.example_pb2 as examples

def load_data():
  # Load data into DataFrame objects.
  columns = ['species', 'petal_length', 'petal_width', 'sepal_length', 'sepal_width']
  df_train = pd.read_csv('data/train.csv', names=columns)
  df_eval = pd.read_csv('data/eval.csv', names=columns)

  return df_train, df_eval

def convert_data(df, path):
  writer = tf.python_io.TFRecordWriter(path)
  for index, row in df.iterrows():
    example = examples.Example()
    features = example.features
    features.feature['species'].bytes_list.value.append(row['species'])
    features.feature['petal_length'].float_list.value.append(row['petal_length'])
    features.feature['petal_width'].float_list.value.append(row['petal_width'])
    features.feature['sepal_length'].float_list.value.append(row['sepal_length'])
    features.feature['sepal_width'].float_list.value.append(row['sepal_width'])

    record = example.SerializeToString()
    writer.write(record)
  writer.close()


def main():
  df_train, df_eval = load_data()
  convert_data(df_train, 'data/train.tfrecord')
  convert_data(df_eval, 'data/eval.tfrecord')


if __name__ == '__main__':
  main()
