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

import os
import collections
import time
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorfx as tfx

class ModelExportManager(object):
  def __init__(self, intermediate_model_folder, final_model_folder, exports_to_keep):
    """Exports last model into final location, keeps last few trained models.

    intermediate_model_folder will contain the last 'exports_to_keep' models
    in folders named with the timestamp they were made. On each model export, 
    the last model is copied into the final model folder. This means the last 
    model is saved into two locations: in final_model_folder and in one of the
    subfolders in intermediate_model_folder.

    intermediate_model_folder or final_model_folder may not contain the other 
    as a child folder.

    Arguments:
      - intermediate_model_folder: path to write all models to
      - final_model_folder: path of latest model
      - exports_to_keep: how many modles should be kept in the
          intermediate_model_folder
    """
    self._intermediate_model_folder = intermediate_model_folder
    self._final_model_folder = final_model_folder
    self._exports_to_keep = exports_to_keep
    self._model_paths = collections.deque()

  def save(self, session, inputs, outputs):
    """Manages calls to tensorfx.prediction.Model.save().

    Exports the graph via tensorfx.prediction.Model.save(). Keeps
    'exports_to_keep' many exports in the intermediate model folder, and copies
    the last model to the final model folder.

    Arguments:
      - session: the TensorFlow session with variables to save.
      - inputs: the list of tensors constituting the input to the prediction graph.
      - outputs: the list of tensors constituting the outputs of the prediction graph.
    """
    new_model_path = os.path.join(self._intermediate_model_folder,
                                  str(time.time()))
    self._model_paths.append(new_model_path)

    tfx.prediction.Model.save(session, new_model_path, inputs, outputs)

    # Remove the old model and copy the lastest into the final_model_folder
    if file_io.is_directory(self._final_model_folder):
      file_io.delete_recursively(self._final_model_folder)
    _recursive_copy(new_model_path, self._final_model_folder)

    # remove old models.
    if len(self._model_paths) > self._exports_to_keep:
      delete_path = self._model_paths.popleft()
      file_io.delete_recursively(delete_path)


def _recursive_copy(src_dir, dest_dir):
  """Copy the contents of src_dir into the folder dest_dir.
  Args:
    src_dir: gsc or local path.
    dest_dir: gcs or local path.
  When called, dest_dir should exist.
  """
  file_io.recursive_create_dir(dest_dir)
  for file_name in file_io.list_directory(src_dir):
    old_path = os.path.join(src_dir, file_name)
    new_path = os.path.join(dest_dir, file_name)

    if file_io.is_directory(old_path):
      _recursive_copy(old_path, new_path)
    else:
      file_io.copy(old_path, new_path, overwrite=True)
