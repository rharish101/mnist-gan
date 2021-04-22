#!/bin/bash -e
location="${1:-"./datasets/MNIST/"}"  # the first argument is where the dataset will be downloaded
if [ -d "$location" ]; then
    echo "Path \"$location\" already exists"
    exit 1
fi

mkdir -p "$location"
pushd "$location" > /dev/null
curl --remote-name-all http://yann.lecun.com/exdb/mnist/{train-images-idx3,train-labels-idx1,t10k-images-idx3,t10k-labels-idx1}-ubyte.gz
gzip -d *.gz
echo "MNIST dataset saved in \"$(pwd)\""
popd > /dev/null
