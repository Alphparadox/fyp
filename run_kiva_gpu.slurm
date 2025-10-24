#!/bin/bash
#SBATCH --job-name=kiva_gpu
#SBATCH --output=../logs/kiva_gpu_%j.out
#SBATCH --error=../logs/kiva_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --exclude=node1
#SBATCH --time=02:00:00

# Activate your own miniconda installation/environment
source ~/miniconda3/bin/activate kiva_env

echo "Running on: $HOSTNAME"
nvidia-smi

# Run the Python script
python fyp.py \
  A.png B.png C.png cand1.png cand2.png cand3.png output_transformed.png
