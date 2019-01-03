#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH -A fallp
#SBATCH -p plgrid
#SBATCH -J RNN_ARG
#SBATCH --output="ARG.out"

module add plgrid/tools/python/3.6.5
module add test/torch

cd $SLURM_SUBMIT_DIR

srun python3.6 ./train.py ARG
