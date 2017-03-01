#/bin/sh

python -m tensorfx.tools.train $1 \
  --module census.train \
  --output /tmp/tensorfx/census \
  --data-train census/data/train.csv \
  --data-eval census/data/eval.csv \
  --data-schema census/data/schema.yaml \
  --data-metadata census/data/metadata.json \
  --data-features census/features.yaml \
  --log-level-tensorflow ERROR \
  --log-level INFO \
  --batch-size 5 \
  --max-steps 1000 \
  --checkpoint-interval-secs 1 \
  --hidden-layers:1 200 \
  --hidden-layers:2 100 \
  --hidden-layers:3 20 \
