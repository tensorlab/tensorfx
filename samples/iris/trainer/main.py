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

# main.py
# Implements the iris classification training job.

import tensorfx as tfx
import tensorfx.models.nn as nn

args = nn.FeedForwardClassificationArguments.parse(parse_job=True)
dataset = tfx.data.CsvDataSet(args.data_schema,
                              train=args.data_train,
                              eval=args.data_eval,
                              metadata=args.data_metadata,
                              features=args.data_features)

classification = nn.FeedForwardClassification(args)

trainer = tfx.training.ModelTrainer()
trainer.train(classification, dataset, args.output)
