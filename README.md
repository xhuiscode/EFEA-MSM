# EFEA-MSM
This is the open source code for paper:Enhanced Facial Expression Analysis via Multi-Scale Polymerization and Cross-Dimensional Modulation.

#Table of Contents
Paper Abstract
Preparation
Running

#Paper Abstract
#Preparation
#FER2013 Dataset
Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data Image Properties: 48 x 48 pixels (2304 bytes) labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.
#CK+ Dataset
The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos, We extracted the last three frames from each sequence in the CK+ dataset, which contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

#Environment
Python 3.7
PyTorch 1.12.1
torchaudio 0.12.1
torchvision 0.13.1
transformers 4.29.2

#Running
To get a quick start, you can run the following command
python main.py

If you are interested in our work, please contact 894190629@shu.edu.cn.
