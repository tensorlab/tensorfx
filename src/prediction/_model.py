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

# _model.py
# Implements the Model base class.

class Model(object):
  """A model provides performs inferences using TensorFlow to produce predictions.

  A model is loaded from a checkpoint that was produced during training.
  """
  def __init__(self, session):
    """Initializes a Model using a TensorFlow session containing an initialized prediction graph.

    Arguments:
      session: The TensorFlow session to use for evaluating inferences.
    """
    self._session = session

  @classmethod
  def load(cls, path, graph_builder=None):
    """Loads a model from a checkpoint.

    Usually the model is loaded from a checkpoint and previously saved prediction graph.
    If a graph builder is specified, it will be used instead. In that case, arguments needed
    to build a graph (to match those used at training time) will be looked from a file
    alongside the saved checkpoint. The graph builder approach is required for graphs that use 
    the py_func op.

    Arguments:
      - path: The location on disk where the checkpoint exists.
      - graph_builder: Optionally, the graph builder to use to produce the prediction graph.
    Returns:
      An initialized Model object that can be used for performing inferences.
    """
    raise NotImplementedError()

  def predict(self, instances):
    """Performs inference to return predictions for the specified instances of data.

    Arguments:
      - instances: either an object, or list of objects each containing feature values.
    """
    raise NotImplementedError()
