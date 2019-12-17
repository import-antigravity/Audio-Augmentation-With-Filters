#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=0-10:00:00
#SBATCH -p gpu -C gpu2080 --gres=gpu:1
#SBATCH --mail-user=grd27@case.edu
#SBATCH --mail-type=ALL
#SBATCH --output=aug_3x.txt
#SBATCH --job-name="aug_3x"

echo "Loading modules..."

module swap intel gcc
module load python/3.6.6
module load cuda/9.0

cd ~/Audio-Augmentation-With-Filters/hpc-scripts
echo "Running python script..."
python train1dcnnrand_aug.py 1 ../models/aug_3x 3
