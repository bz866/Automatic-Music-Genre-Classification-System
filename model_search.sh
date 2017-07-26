#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=song_tagger
#SBATCH --time=3:00:00

module purge
module load python/intel/2.7.12
module load scikit-learn/intel/0.18.1
module load pytorch/intel/20170125
module load torchvision/0.1.7

cd /home/$USER/DS1003_MLProject

python model_search.py
