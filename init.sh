#!/bin/sh

ln -s src tensorfx

export REPO=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO:$REPO/samples:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

# simulate script entrypoint created in setup for use in development use-cases.
echo $PATH | grep -q "/tmp/tensorfx/bin"
if [[ $? -ne 0 ]]; then
  export PATH=/tmp/tensorfx/bin:$PATH
  mkdir -p /tmp/tensorfx/bin
  echo "#! /bin/sh
  python -m tensorfx.tools.tfx \$@
  " > /tmp/tensorfx/bin/tfx
  chmod u+x /tmp/tensorfx/bin/tfx
fi

# Optionally install python packages
if [ "$1" == "pip" ]; then
  pip install -r requirements.txt
fi

