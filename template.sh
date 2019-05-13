#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH -A fallp2
#SBATCH -p plgrid
#SBATCH --output="fresh.out"

module add plgrid/tools/python/3.6.5
module add test/torch
module add plgrid/libs/python-scipy/1.0.1-python-3.6

cd $SLURM_SUBMIT_DIR

srun python3.6 ./train.py
