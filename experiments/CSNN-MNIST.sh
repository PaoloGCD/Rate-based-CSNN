#!/bin/bash
mkdir ./results/CSNN-MNIST-400
python ./code/CSNN/training.py ./params/CSNN-MNIST-400.xml ./results/CSNN-MNIST-400/
python ./code/CSNN/labeling.py ./params/CSNN-MNIST-400.xml ./results/CSNN-MNIST-400/
python ./code/CSNN/testing.py ./params/CSNN-MNIST-400.xml ./results/CSNN-MNIST-400/