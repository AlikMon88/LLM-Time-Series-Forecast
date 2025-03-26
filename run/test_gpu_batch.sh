#!/bin/bash
#SBATCH --job-name=m2-gwen-test        # Meaningful job name
#SBATCH --output=/user/am3353/am3353/log/output_test.log
#SBATCH --error=/user/am3353/am3353/log/error_test.log
#SBATCH --time=00:10:00                # Run for 10 minutes
#SBATCH --partition=ampere              # GPU partition
#SBATCH --gres=gpu:1                    # Request only 1 GPU
#SBATCH --ntasks=1                       # Use only 1 task
#SBATCH --cpus-per-task=1                # Use 1 CPU per task
#SBATCH --mem=8G                         # Request 8GB RAM
#SBATCH -A MPHIL-DIS-SL2-GPU             # Your project account

# Load modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment
source /user/am3353/am3353/m2-env/bin/activate

# Debug: Check Python & CUDA
python --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device Count:', torch.cuda.device_count())"

# Debug: Run a simple test
python -c "print('Hello from inside Slurm!')"

# Run training script with logging
python lora_train.py > /user/am3353/am3353/log/train_output.log 2>&1
