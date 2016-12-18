#!/usr/bin/python
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

# To publish to PyPi, use:
# python setup.py bdist_wheel upload -r pypi

import datetime
import setuptools

version = '0.1.' + datetime.datetime.now().strftime("%y%m%d%H%M")

setuptools.setup(
  name='tensorfx',
  version=version,
  namespace_packages=['tensorfx'],
  packages=[
    'tensorfx.models',
    'tensorfx.data',
    'tensorfx.training',
    'tensorfx.serving',
    'tensorfx.toolbox'
  ],
  install_requires=[
  ],
  author='Nikhil Kothari',
  author_email='nikhilk@twitter',
  url='https://github.com/TensorLab/tensorfx',
  license="Apache Software License",
  description='TensorFX Framework',
  keywords=[
    'TensorLab',
    'TensorFlow',
    'Machine Learning',
    'Deep Learning',
    'Google'
  ],
  classifiers=[
  ]
)

