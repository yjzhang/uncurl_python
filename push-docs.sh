#!/bin/bash

# run this from the master branch to build/push documentation

cd docs
make html
git add _build
git commit

cd ..

git subtree push --prefix docs/_build/html origin gh-pages

