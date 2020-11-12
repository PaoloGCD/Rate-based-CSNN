#!/bin/bash
mkdir ./results/CSNN-FashionMNIST-400
python ./code/CSNN/training.py ./params/CSNN-FashionMNIST-400.xml ./results/CSNN-FashionMNIST-400/
python ./code/CSNN/labeling.py ./params/CSNN-FashionMNIST-400.xml ./results/CSNN-FashionMNIST-400/
python ./code/CSNN/testing.py ./params/CSNN-FashionMNIST-400.xml ./results/CSNN-FashionMNIST-400/