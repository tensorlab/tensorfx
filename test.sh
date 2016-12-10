#!/bin/sh
set -e

# Clean out tests
mkdir -p build/tests
rm -rf build/tests

# Copy test sources
cp -rf tests build/tests

# Run tests
pushd build/tests > /dev/null
python main.py
popd > /dev/null

echo 'Tests completed successfully!'

