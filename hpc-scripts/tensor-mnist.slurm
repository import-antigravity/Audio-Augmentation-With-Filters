#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00 --mem=5gb
#SBATCH -p gpu -C gpuk40 --gres=gpu:1 --nodelist=gput026


cp -r mnist_softmax.py $PFSDIR
cd $PFSDIR

module load singularity

echo "Run Tensorflow Convolutional Neural Network (CNN) Job:"
singularity exec -B --nv $TENSORFLOW python mnist_softmax.py

cp -ru * $SLURM_SUBMIT_DIR
