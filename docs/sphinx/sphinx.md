# How to build Sphinx documentation

## Install dependencies
pip install sphinx-markdown-builder

## Go to docs/sphinx
sphinx-apidoc -o source/ ../../
sphinx-build -b markdown source/ sphinx_doc/

