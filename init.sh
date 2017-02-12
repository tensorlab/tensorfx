#!/bin/sh

ln -s src tensorfx

export REPO=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:$REPO/src:$REPO/samples
export PYTHONDONTWRITEBYTECODE=1

# Optionally install python packages
if [ "$1" == "pip" ]; then
  pip install -r requirements.txt
fi

