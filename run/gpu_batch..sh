#!/bin/bash
#SBATCH --job-name=m2-gwen-lora         # Change this to a meaningful name
#SBATCH --output=/rds/am3353/am3353/log/output.log   # Change this path
#SBATCH --error=/rds/am3353/am3353/log/error.log     # Change this path
#SBATCH --time=02:00:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere            # GPU partition
#SBATCH --gres=gpu:4                   # Request 1 GPU (adjust if needed)
#SBATCH -A MPHIL-DIS-SL2-GPU        # Your project account (check with `groups` command)
#SBATCH --cpus-per-task=4              # Number of CPU cores per GPU
#SBATCH --mem=16G                      # Adjust memory based on your workload

# Load necessary modules (adjust based on your environment)
module load python/3.10  # Load Python module (change version if needed)
module load cuda/11.7   # Load CUDA (change version if needed)

# Activate virtual environment if needed
source /user/am3353/am3353/m2-env/bin/activate

# Run your script
python my_script.py --arg1 value1 --arg2 value2
