#!/bin/bash

## 1.(a) load the CC python3.7 module & required libs:
module load python/3.7
module load cuda/10.0 cudnn
module load hdf5

## -or- 1.(b) choose to load python3.8 instead:
#module load python/3.8
#module load cuda/10.1.243 cudnn
#module load hdf5


## 2. Create a path for venv
mkdir -p ~/work/venv
# or use the project path to share virtualenv:
#mkdir -p ~/project/venv

## 2.2. Confirm python version & paths (to make sure modules loaded ok:
module list
which python
which pip
python --version

## 3.1. Create a new virtual environment named tf-py37:
virtualenv ~/work/venv/tf-py37

## 3.2. activate env:
source ~/work/venv/tf-py37/bin/activate  

## 3.3. run pip to install tf 1.x:
pip install tensorflow-gpu==1.15.0

## See also table: https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible
# Trixie presently supports cuda versions: 10.0.x and 10.1.x:
#   - TF 1 works with cuda 10.0.130
#   - TF 2 works with cuda 10.1.243
