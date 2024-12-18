#!/bin/bash

#SBATCH --job-name=pop_dce_1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=18G
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --reservation=gpu
#SBATCH --nice=19

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# show CPU device
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# modules
module load devel/cuda/9.1

# conda
echo "activating conda"
cd ~/scratch/DCE-NET/pop/DCE-NET-1
source /scratch/pschouten/virtualenv/env1/bin/activate
conda activate /scratch/pschouten/virtualenv/env1
echo "conda activated"

# python
echo "starting python"
nice python main.py
echo "finished python"

# mail
echo "run time mailed"


