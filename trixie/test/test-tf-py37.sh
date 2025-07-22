#!/bin/bash

# Specify the partition of the cluster to run on (Typically TrixieMain)
#SBATCH --partition=TrixieMain
# Add your project account code using -A or --account
#SBATCH --account ai4d
# Specify the time allocated to the job. Max 12 hours on TrixieMain queue.
#SBATCH --time=12:00:00
# Request GPUs for the job. In this case 1 GPU
#SBATCH --gres=gpu:1
# Print out the hostname that the jobs is running on
hostname
# Run nvidia-smi to ensure that the job sees the GPUs
/usr/bin/nvidia-smi

module load python/3.7
module load cuda/10.0 cudnn
module load hdf5

source ~/work/venv/tf-py37/bin/activate  

# Launch our test tensorflow python file
python test-tf-py37.py
