#!/bin/bash

# Slurm job configuration
#SBATCH --job-name="test_subspace_steering"
#SBATCH --account=ax74              # project/group account
#SBATCH --partition=gpu             # The GPU partition (check cluster docs for the right name)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=04:00:00               # Request 4 hours of runtime (hh:mm:ss)
#SBATCH --mem=64G                   # Request 64 GB of memory
#SBATCH --cpus-per-task=8      # Add CPU cores for parallel processing
#SBATCH --output=test_subspace_steering_%j.log    # Save output to a log file

# --- Job Commands ---

# 1. Change to your project directory
cd /home/pdoo0004/AlphaSteer

# 2. Load the necessary modules
module load anaconda/2024.02-1

# 3. Initialize Conda in the script's shell
source ~/.bashrc

# 4. Activate your Conda environment
conda activate alphasteer

# 5. Run the subspace steering tests
echo "Starting subspace steering inference test..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Test with alpaca_eval config first (smaller dataset for quick validation)
echo "Testing with alpaca_eval config..."
python src/generate_subspace_response.py --config_path config/llama3.1_subspace/alpaca_eval.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "=== alpaca_eval test PASSED ==="
    echo ""

    # If alpaca_eval works, test jailbroken config
    echo "Testing with jailbroken config..."
    python src/generate_subspace_response.py --config_path config/llama3.1_subspace/jailbroken.yaml

    if [ $? -eq 0 ]; then
        echo ""
        echo "=== jailbroken test PASSED ==="
        echo "=== All tests completed successfully ==="
    else
        echo ""
        echo "=== jailbroken test FAILED ==="
        exit 1
    fi
else
    echo ""
    echo "=== alpaca_eval test FAILED ==="
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "Subspace steering pipeline test finished."