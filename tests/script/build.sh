#!/bin/bash

pip uninstall -y llmdnn
cd ../../build/ || exit
make -j 20
cd - || exit
cd ext || exit
python setup.py clean --all install
