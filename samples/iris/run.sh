#!/bin/sh

python -m tensorfx.tools.train $1 \
  --module iris.trainer.main \
  --output /tmp/tensorfx/iris/csv \
  --data-train iris/data/train.csv \
  --data-eval iris/data/eval.csv \
  --data-schema iris/data/schema.yaml \
  --data-metadata iris/data/metadata.json \
  --data-features iris/features.yaml \
  --log-level-tensorflow ERROR \
  --log-level INFO \
  --batch-size 5 \
  --max-steps 2000 \
  --checkpoint-interval-secs 1 \
  --hidden-layers:1 20 \
  --hidden-layers:2 10 \
