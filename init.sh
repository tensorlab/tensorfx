#!/bin/sh

ln -s src tensorfx

export REPO=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO:$REPO/samples:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

# Setup aliases to simulate console entrypoints created in setup for use in
# development use-cases.
shopt -s expand_aliases
alias tfx="python -m tensorfx.tools.tfx"

# Optionally install python packages
if [ "$1" == "pip" ]; then
  pip install -r requirements.txt
fi

