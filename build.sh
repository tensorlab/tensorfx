#!/bin/sh
set -e

# Create the build directory with inputs (sources) and outputs (binaries).
# Ensure it is empty, to build from clean state.
mkdir -p build
rm -rf build
mkdir -p build

# Copy source files
cp requirements.txt build
cp setup.py build
cp -r src build/tensorfx

# Generate the README expected by PyPI from original markdown
curl --silent http://c.docverter.com/convert \
  -F from=markdown \
  -F to=rst \
  -F input_files[]=@README.md > build/README.rst

# Finally, build
pushd build > /dev/null
python setup.py sdist > setup.log
popd > /dev/null

echo 'Build completed successfully!'

