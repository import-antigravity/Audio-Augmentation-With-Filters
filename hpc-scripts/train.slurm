#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=0-10:00:00
#SBATCH -C gpu2080 --gres=gpu:1
#SBATCH --mail-user=grd27@case.edu
#SBATCH --mail-type=ALL
#SBATCH --output=cnn_rand32_filter10.txt
#SBATCH --job-name="32filter10"

echo "Loading modules..."

module swap intel gcc
module load python/3.6.6
module load cuda/9.0

cd ~/Audio-Augmentation-With-Filters/hpc-scripts
echo "Running python script..."
python cnn_rand32k.py 10 1
