import json
import numpy as np
import pandas as pd
import tensorfx as tfx

NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
         'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss',
         'hours_per_week', 'native_country', 'income_bracket']

TYPES = [np.int32, np.str, np.int32, np.str, np.int32, np.str,
         np.str, np.str, np.str, np.str, np.int32, np.int32,
         np.int32, np.str, np.str]

def load_data(path):
  # Load data, while also stripping leading spaces and converting to ? to missing values
  df = pd.read_csv(path, names=NAMES, skipinitialspace=True, na_values=['?'])

  for name, dtype in zip(NAMES, TYPES):
    if dtype == np.str:
      df[name] = df[name].astype('category')

  # Drop useless/redundant columns
  df.drop('fnlwgt', 1, inplace=True)
  df.drop('education', 1, inplace=True)

  # Order columns so that the target label is the first column
  cols = df.columns.tolist()
  cols = cols[-1:] + cols[0:-1]
  df = df[cols]

  return df


def create_schema(df, path):
  schema_fields = []
  for name, dtype in zip(df.columns, df.dtypes):
    if type(dtype) == pd.types.dtypes.CategoricalDtype:
      field_type = tfx.data.SchemaFieldType.discrete
    elif dtype == np.int64 or dtype == np.int32 or dtype == np.float64 or dtype == np.float32:
      field_type = tfx.data.SchemaFieldType.numeric

    field = tfx.data.SchemaField(name, field_type)
    schema_fields.append(field)

  schema = tfx.data.Schema(schema_fields)
  with open(path, 'w') as f:
    f.write(schema.format())


def create_metadata(df, path):
  metadata = {}
  for name, dtype in zip(df.columns, df.dtypes):
    md = {}
    if type(dtype) == pd.types.dtypes.CategoricalDtype:
      entries = list(df[name].unique())
      if np.nan in entries:
        entries.remove(np.nan)
      md['entries'] = sorted(entries)
    elif dtype in (np.int32, np.int64, np.float32, np.float64):
      for stat, stat_value in df[name].describe().iteritems():
        if stat == 'min':
          md['min'] = stat_value
        if stat == 'max':
          md['max'] = stat_value

    metadata[name] = md

  with open(path, 'w') as f:
    f.write(json.dumps(metadata, separators=(',',':')))


# Load train and eval data
df_train = load_data('census/data/raw/train.csv')
df_eval = load_data('census/data/raw/eval.csv')

# Produce processed data
df_train.to_csv('census/data/train.csv', header=False, index=False)
df_eval.to_csv('census/data/eval.csv', header=False, index=False)

# Produce schema
create_schema(df_train, 'census/data/schema.yaml')

# Produce metadata
create_metadata(df_train, 'census/data/metadata.json')
