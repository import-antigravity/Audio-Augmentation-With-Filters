#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00 --mem=32gb
#SBATCH -p gpu -C gpu2080 --gres=gpu:2
#SBATCH --mail-user=grd27@case.edu
#SBATCH --output=~/aug_2x.txt

module swap intel gcc
module load python/3.6.6
module load cuda/9.0

cd ~/Audio-Augmentation-With-Filters/hpc-scripts
python train1dcnnrand_aug.py 2 ../models/aug_2x 2
