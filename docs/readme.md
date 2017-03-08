# Reference Documentation

## Setup
Docs are built using Sphinx, and use the 'read-the-docs' theme.

    pip install sphinx sphinx_rtd_theme sphinxcontrib-napoleon

## Build
Docs are built into the ../../tensorfx-docs/html directory, which is expected to be a clone of the
gh-pages branch of this repository.

    mkdir -p ../../tensorfx-docs
    make html
