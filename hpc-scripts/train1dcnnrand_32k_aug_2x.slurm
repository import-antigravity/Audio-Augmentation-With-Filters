#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=0-10:00:00
#SBATCH -p gpu -C gpu2080 --gres=gpu:1
#SBATCH --mail-user=grd27@case.edu
#SBATCH --mail-type=ALL
#SBATCH --output=aug_2x.txt
#SBATCH --job-name="aug_2x"

echo "Loading modules..."

module swap intel gcc
module load python/3.6.6
module load cuda/9.0

echo "Copying to PFSDIR..."
echo $PFSDIR
cp -r ~/Audio-Augmentation-With-Filters $PFSDIR
cd $PFSDIR/Audio-Augmentation-With-Filters/hpc-scripts

echo "Running python script..."
python train1dcnnrand_aug_gen.py 1 aug_2x/ 2

cp -r aug_2x ~/Audio-Augmentation-With-Filters/models/
