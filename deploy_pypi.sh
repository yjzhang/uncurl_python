#!/bin/sh

# before running this:
# 1. increase version number in setup.py, git commit
# 2. git tag v<new version>
# 3. git push --tags

# delete existing dists
rm dist/*.tar.gz
rm dist/*.whl

# create a source distribution
python3 setup.py sdist

# create wheels - note: this can't be uploaded onto pypi
python3 setup.py bdist_wheel

# upload
twine upload dist/uncurl_seq-*.tar.gz --verbose

# TODO: how to upload built wheels? This requires the 'manylinux1' platform tag?
