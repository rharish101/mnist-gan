#!/bin/bash -e

# SPDX-FileCopyrightText: 2019 Harish Rajagopal <harish.rajagopals@gmail.com>
#
# SPDX-License-Identifier: MIT

location="${1:-"./datasets/MNIST/"}"  # the first argument is where the dataset will be downloaded
mkdir -p "$location"
pushd "$location" > /dev/null
curl --remote-name-all http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
gzip -d *.gz
echo "MNIST dataset saved in \"$(pwd)\""
popd > /dev/null
