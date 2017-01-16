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

# _ds_df.py
# Implementation of DataFrameDataSource.

from ._dataset import DataSource

class DataFrameDataSource(DataSource):
  """A DataSource representing a Pandas DataFrame.

  This class is useful for working with local/in-memory data.
  """
  def __init__(self, name, df):
    """Initializes an instance of a DataFrameDataSource with the specified Pandas DataFrame.

    Arguments:
      name: the name of the DataSource.
      df: the DataFrame instance to use.
    """
    super(DataFrameDataSource, self).__init__(name)
    self._df = df
