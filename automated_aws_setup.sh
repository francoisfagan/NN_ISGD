#!/bin/bash
mkdir NN_ISGD
mkdir NN_ISGD/results
mkdir NN_ISGD/data
sudo yum update
sudo yum install htop

# tmux new -s e
# cd NN_ISGD
# source activate pytorch_p36
# pip install -U scikit-learn
# pip install scipy
# python pipeline.py 'JSB Chorales'
# tmux a -t e

# pip install Cython

# Move over all python files
# Move over "hyperparameter_lists" and then "data" directories

# To monitor the gpu use: watch -n 1 nvidia-smi