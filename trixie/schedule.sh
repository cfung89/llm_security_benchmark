#! /bin/bash

#SBATCH --job-name=LLM_Bench
#SBATCH --output=slurm_%j.out
# #SBATCH --error=slurm_%j.err

#SBATCH --partition=TrixieMain
#SBATCH --account=dt-cyber

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

# Loading environment
echo "Loading environment..."
module load python/3.12.1-gcc-11.3.1-7tuhjhr
source ~/work/.venv/bin/activate

# Checks
echo -e "Checks...\n"
echo -e "$(hostname)\n"
echo -e "$(which python)\n"
echo -e "$(/usr/bin/nvidia-smi)\n"

# Jobs
echo -e "\nRunning jobs...\n"
inspect eval inspect_evals/gdm_intercode_ctf --model vllm/mistralai/Mistral-Large-Instruct-2411
inspect eval inspect_evals/cybench --model vllm/mistralai/Mistral-Large-Instruct-2411

# Example
# python3 ~/work/test.py
