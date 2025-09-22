#!/bin/bash

# Slurm job configuration
#SBATCH --job-name="subspaceSteer"    
#SBATCH --account=ax74              # project/group account
#SBATCH --partition=fit             # The GPU partition (check cluster docs for the right name)
#SBATCH --gres=gpu:A100:1                   # Request 1 GPU
#SBATCH --qos=fitq
#SBATCH --time=08:00:00               # Request 4 hours of runtime (hh:mm:ss)
#SBATCH --mem=256G                   # Request 64 GB of memory
#SBATCH --cpus-per-task=8      # Add CPU cores for parallel processing
#SBATCH --output=subspaceSteer_%j.log    # Save output to a log file

# --- Job Commands ---

# 1. Change to your project directory
cd /home/pdoo0004/AlphaSteer

# 2. Load the necessary modules
module load anaconda/2024.02-1

# 3. Initialize Conda in the script's shell
source ~/.bashrc

# 4. Activate your Conda environment
conda activate alphasteerv3

# 5. Run the main script
echo "Starting SubspaceSteer script..."
bash ./scripts/subspace_steering.sh
echo "Script finished."
