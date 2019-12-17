#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00 --mem=32gb
#SBATCH -p gpu -C gpup100 --gres=gpu:2
#SBATCH --mail-user=grd27@case.edu

module swap intel gcc
module load python/3.6.6
module load cuda/9.0

cd ~/Audio-Augmentation-With-Filters/hpc-scripts
python train1dcnnrand_aug.py 2 ../models/aug_3x 3
