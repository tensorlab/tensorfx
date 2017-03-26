#!/bin/sh

python -m tensorfx.tools.tfx train $1 \
  --module trainer.examples \
  --output /tmp/tensorfx/iris/examples \
  --data-train data/train.tfrecord \
  --data-eval data/eval.tfrecord \
  --data-schema data/schema.yaml \
  --data-metadata data/metadata.json \
  --data-features trainer/features.yaml \
  --log-level-tensorflow ERROR \
  --log-level INFO \
  --batch-size 5 \
  --max-steps 2000 \
  --checkpoint-interval-secs 1 \
  --hidden-layers:1 20 \
  --hidden-layers:2 10 \

