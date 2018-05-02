#!/bin/bash
mkdir NN_ISGD
mkdir NN_ISGD/results
sudo yum update
sudo yum install htop
source activate pytorch_p36
# tmux new -s experiment1

# pip install Cython

# Move over all python files
# Move over "hyperparameter_lists" and then "data" directories

# To monitor the gpu use: watch -n 1 nvidia-smi