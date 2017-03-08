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

import setuptools

with open('tensorfx/_version.py') as vf:
  exec(vf.read())

with open('requirements.txt') as rf:
  dependencies = rf.readlines()
  dependencies = map(lambda d: d.strip(), dependencies)
  dependencies = filter(lambda d: d and not d.startswith('#'), dependencies)

setuptools.setup(
  name='tensorfx',
  version=__version__,
  packages=[
    'tensorfx',
    'tensorfx.data',
    'tensorfx.training',
    'tensorfx.prediction',
    'tensorfx.tools',
    'tensorfx.models',
    'tensorfx.models.nn'
  ],
  install_requires=dependencies,
  author='Nikhil Kothari',
  author_email='nikhilk@twitter',
  url='https://github.com/TensorLab/tensorfx',
  license="Apache Software License",
  description='TensorFX Framework for training and serving machine learning models with TensorFlow',
  keywords=[
    'TensorLab',
    'TensorFlow',
    'Machine Learning',
    'Deep Learning',
    'Google'
  ],
  classifiers=[
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'Development Status :: 3 - Alpha',
    'Environment :: Other Environment',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License'
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  data_files=[('.', ['requirements.txt'])]
)
