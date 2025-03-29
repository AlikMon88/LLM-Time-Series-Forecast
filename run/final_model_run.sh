#!/bin/bash
#SBATCH --job-name=m2-gwen-lora         # Change this to a meaningful name
#SBATCH --output=/home/am3353/am3353/log/output.log   # Change this path
#SBATCH --error=/home/am3353/am3353/log/error.log     # Change this path
#SBATCH --time=02:00:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere            # GPU partition
#SBATCH --gres=gpu:4                   # Request 1 GPU (adjust if needed)
#SBATCH -A MPHIL-DIS-SL2-GPU        # Your project account (check with `groups` command)

module load python/3.9.12
module load cuda/11.8

source /home/am3353/am3353/m2-env/bin/activate

# Run your script
python src/final_model.py
