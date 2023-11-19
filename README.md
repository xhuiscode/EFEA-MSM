# EFEA-MSM
This is the open source code for paper:Enhanced Facial Expression Analysis via Multi-Scale Polymerization and Cross-Dimensional Modulation.

#Table of Contents
Paper Abstract
Preparation
Running
# Paper Abstract
Facial expressions are the most crucial cues in human emotional intention perception. With the rapid development of artificial intelligence, automatic facial expression recognition plays a key role in human-computer interaction. Addressing the challenge of insufficiently detailed fusion of facial expression features in existing models, this paper proposes a facial expression recognition model based on an embedded cross-dimensional space modulator and a multi-scale channel aggregator. This model employs the MobileNetV2 network embedded with a cross-dimensional space modulator and a multi-scale channel aggregator to facilitate cross-dimensional feature interaction, capture information mappings in different subspaces, and learn the channel weights of multi-scale feature maps. It then combines these with a fully connected layer for multi-dimensional feature
fusion and expression classification. The proposed model achieves accuracy rates of 75.01% and 98.97% on the public datasets FER2013 and CK+, respectively, demonstrating state-of-the-art results compared to other advanced networks.The experimental results indicate that the proposed method effectively guides feature fusion, enhancing the accuracy and generalizability of facial expression recognition. It is particularly suitable for scenarios like online teaching, criminal investigations and interrogations, and medical diagnosis. The code for this study is available at https://github.com/xhuiscode/EFEA-MSM.
# Preparation
# FER2013 Dataset
Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data Image Properties: 48 x 48 pixels (2304 bytes) labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.
# CK+ Dataset
The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos, We extracted the last three frames from each sequence in the CK+ dataset, which contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

# Environment
Python 3.7
PyTorch 1.12.1
torchaudio 0.12.1
torchvision 0.13.1
transformers 4.29.2

# Running
To get a quick start, you can run the following command
python main.py

If you are interested in our work, please contact 894190629@shu.edu.cn.
