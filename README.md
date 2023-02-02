# 1、paper:Spatio-Temporal SiamFC: Per-Sequence Visual Tracking with Siamese 3D Convolutional Networks

# 2、Installation
## 2.1. create conda virtual env.
conda create -n STSiamFC python=3.6

## 2.2. activate conda virtual env.
conda activate STSiamFC

## 2.3. install pytorch, reference url: https://pytorch.org.
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

## 2.4. install other dependent packages.
conda install numpy matplotlib pillow opencv-python

# 3、Train tracker
## 3.1. Preprocess the dataset(eg:IMAGENET2015)
pthon -u createdataset.py

## 3.2. Start training
pyhon main.py 

# 4、Test tracker

# 5、Eval tracker
