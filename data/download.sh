#!/usr/bin/env bash

curl --remote-name-all \
    https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz \
    https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz \
    https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz \
    https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
gunzip *-ubyte.gz
