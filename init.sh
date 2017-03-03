#!/bin/sh

ln -s src tensorfx

export REPO=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO:$REPO/samples:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

# Optionally install python packages
if [ "$1" == "pip" ]; then
  pip install -r requirements.txt
fi

