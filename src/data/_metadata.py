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

# _metadata.py
# Implementation of Metadata.

import ujson


class Metadata(object):
  """This class encapsulates metadata for individual fields within a dataset.

  Metadata is key'ed by individual field names, and is represented as key/value pairs, specific
  to the type of the field, and the analysis performed to generate the metadata.
  """
  def __init__(self, md):
    """Initializes an instance of a Metadata object.

    Arguments:
      md: the metadata map key'ed by field names.
    """
    self._md = md

  @staticmethod
  def parse(metadata):
    """Parses a Metadata instance from a JSON specification.

    Arguments:
      metadata: The metadata to parse.
    Returns:
      A Metadata instance.
    """
    md = ujson.loads(metadata)
    return Metadata(md)

  def __getitem__(self, index):
    """Retrieves the metadata of the specified field by name.

    Arguments:
      index: the name of the field whose metadata is to be retrieved.
    Returns:
      The metadata dictionary for the specified field, or an empty dictionary.
    """
    return self._md.get(index, {})

  def __len__(self):
    """Retrieves the number of Features defined.
    """
    return len(self._md)
