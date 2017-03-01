# Samples

## Iris

This is the 'Hello World' of machine learning. This demonstrates two ways of
using TensorFX.

* Using in-memory (DataFrame) training and evaluation data. To run:

      python iris/run.py

* Using file-based training and evaluation data, as well as schema, metadata,
  and features declared in external YAML or JSON files. To run:

      iris/run.sh

  Which invokes the train tool from TensorFX to launch a python module as a
  trainer process, as follows:

      python -m tensorfx.tools.train \
        --module iris.trainer.main --output [output path] \
        ... trainer specific argumenrts


