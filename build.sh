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


# Check for setuptools >30. Otherwise, tar file will not have requirements.txt
# pip list --format=columns  (list all versions)
# grep 'setuptools' (gets setuptools x.y.z)
# awk  '{print $2}' (gets x.y.z)
# awk -F. '{print $1}' (gets x)
setuptools_version=`pip list --format=columns | grep 'setuptools' | awk  '{print $2}' | awk -F. '{print $1}'`
if [[ -z $setuptools_version || $setuptools_version -lt 30 ]]; then
  echo 'setuptools version 30 or higher is required.'
  echo "This system has version ${setuptools_version}"
  echo 'First upgrade: pip install --upgrade setuptools'
  exit 1
fi


# Finally, build
pushd build > /dev/null
python setup.py sdist > setup.log
popd > /dev/null

echo 'Build completed successfully!'


# Copy over samples and turn them into a module
cp -r samples build/samples
touch build/samples/__init__.py

echo 'Samples copied successfully!'


# Copy over tests
cp -r tests build/tests

echo 'Tests copied successfully!'


# Optionally run tests
if [ "$1" == "test" ]; then
  pushd build/tests > /dev/null
  python main.py
  popd > /dev/null

  echo 'Tests completed'
fi
