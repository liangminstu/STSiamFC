# 1、paper:Spatio-Temporal SiamFC: Per-Sequence Visual Tracking with Siamese 3D Convolutional Networks

## 2、Installation
### 2.1. create conda virtual env.
```
conda create -n STSiamFC python=3.6
```

### 2.2. activate conda virtual env.
```
conda activate STSiamFC
```

### 2.3. install pytorch, reference url: https://pytorch.org.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### 2.4. install other dependent packages.
```
conda install numpy matplotlib pillow opencv-python
```

## 3、Train tracker
### 3.1. Preprocess the dataset(eg:IMAGENET2015)
```
cd data_process
pthon -u createdataset.py
```

### 3.2. Start training
```
pyhon main.py 
```

## 4、Test tracker
```
python -u Track/testVOT.py                           
```

## 5、Eval tracker
Testing results are saved in results.zip
```
python /eval/eval.py              
        --tracker_path ./results        \ # result path
        --dataset VOT2018               \ # dataset name
        --num 1                         \ # number thread to eval
        --tracker_prefix 'checkpoint'     # tracker_name_prefix
```
## References
```
[1] VOT python toolkit: https://github.com/StrangerZhang/pysot-toolkit
```
