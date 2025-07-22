#! /bin/bash

#SBATCH --job-name=LLM_Bench

#SBATCH --partition=TrixieMain
#SBATCH --acount=dt-

#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nodes=1

#SBATCH --requeue
#SBATCH --signal=B:USR1@30

# Requeuing
function _requeue {
   echo "BASH - trapping signal 10 - requeueing $SLURM_JOBID"
   date
   scontrol requeue $SLURM_JOBID
}

if [[ -n "$SLURM_JOBID" ]]; then
   trap _requeue USR1
fi

# Checks
hostname
/usr/bin/nvidia-smi

# Job

module load python/3.12.1-gcc-11.3.1-7tuhjhr

source ~/work/.venv/bin/activate

python3 ~/work/test.py
