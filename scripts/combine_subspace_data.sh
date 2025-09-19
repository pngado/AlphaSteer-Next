#!/bin/bash

# Script to combine learned subspaces and steering matrices for subspace steering
# This creates the combined data file expected by our SubspaceAlphaLlamaForCausalLM model

# Configuration
MODEL_NAME=llama3.1

# Input paths (adjust these if your file names are different)
LEARNED_SUBSPACES_PATH="data/subspaces/learned_subspaces_${MODEL_NAME}.pkl"
STEERING_MATRICES_PATH="data/steering_matrices/steering_matrices_${MODEL_NAME}.pkl"

# Output path (matches what the config files expect)
OUTPUT_PATH="data/subspace_steering/subspace_data_${MODEL_NAME}.pkl"

echo "=== Combining Subspace Data for $MODEL_NAME ==="
echo "Input files:"
echo "  Learned subspaces: $LEARNED_SUBSPACES_PATH"
echo "  Steering matrices: $STEERING_MATRICES_PATH"
echo "Output file:"
echo "  Combined data: $OUTPUT_PATH"
echo ""

# Check if input files exist
if [ ! -f "$LEARNED_SUBSPACES_PATH" ]; then
    echo "Error: Learned subspaces file not found: $LEARNED_SUBSPACES_PATH"
    echo "Please run learn_benign_subspaces.py first"
    exit 1
fi

if [ ! -f "$STEERING_MATRICES_PATH" ]; then
    echo "Error: Steering matrices file not found: $STEERING_MATRICES_PATH"
    echo "Please run calc_subspace_steering_matrix.py first"
    exit 1
fi

# Run the combination script
python combine_subspace_data.py \
    --learned_subspaces_path "$LEARNED_SUBSPACES_PATH" \
    --steering_matrices_path "$STEERING_MATRICES_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Success! ==="
    echo "Combined subspace data created at: $OUTPUT_PATH"
    echo "Ready to use with subspace steering configs in config/llama3.1_subspace/"
    echo ""
    echo "Next steps:"
    echo "1. Test the pipeline: python src/generate_subspace_response.py --config_path config/llama3.1_subspace/alpaca_eval.yaml"
    echo "2. Run all datasets: bash scripts/generate_subspace.sh"
else
    echo ""
    echo "=== Error ==="
    echo "Failed to combine subspace data. Check the error messages above."
    exit 1
fi