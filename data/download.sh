#!/usr/bin/env bash

curl --remote-name-all \
    https://raw.githubusercontent.com/chitraksen/mnist-data/main/train-images-idx3-ubyte.gz \
    https://raw.githubusercontent.com/chitraksen/mnist-data/main/train-labels-idx1-ubyte.gz \
    https://raw.githubusercontent.com/chitraksen/mnist-data/main/t10k-images-idx3-ubyte.gz  \
    https://raw.githubusercontent.com/chitraksen/mnist-data/main/t10k-labels-idx1-ubyte.gz
gunzip *-ubyte.gz
