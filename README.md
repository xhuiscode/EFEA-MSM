# EFEA-MSM
This is the open source code for paper:Enhanced Facial Expression Analysis via Multi-Scale Polymerization and Cross-Dimensional Modulation.

Table of Contents
Paper Abstract
Preparation
Running
Training

Paper Abstract


Preparation
Datasets
As mentioned in our paper, in order to train our model, you need to download the CK+ and FER2013 dataset here.

Environment
Python 3.7
PyTorch 1.12.1
torchaudio 0.12.1
torchvision 0.13.1
transformers 4.29.2

Running
To get a quick start, you can run the following command
python main.py

Training
Training the Facial Expression Recognition Model

# Single machine, single GPU training
CUDA_VISIBLE_DEVICES =0
python train.py
# Single machine, multi -GPU training
CUDA_VISIBLE_DEVICES =0,1 torchrun -- standalone -- nnodes =1
-- nproc_per_node =2
python train.py

If you are interested in our work, please contact 894190629@shu.edu.cn.
