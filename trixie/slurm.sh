#! /bin/bash

#SBATCH --partition=TrixieMain
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@30

function _requeue {
   echo "BASH - trapping signal 10 - requeueing $SLURM_JOBID"
   date
   scontrol requeue $SLURM_JOBID
}

if [[ -n "$SLURM_JOBID" ]]; then
   trap _requeue USR1
fi

# Job

wait
