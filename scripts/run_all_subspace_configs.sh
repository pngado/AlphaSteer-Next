#!/bin/bash

# Script to run generate_subspace_response.py for all YAML configs in config/llama3.1_subspace/

CONFIG_DIR="config/llama3.1_subspace"
SCRIPT="src/generate_subspace_response.py"

echo "Running subspace response generation for all configs in $CONFIG_DIR"

for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config_file" ]; then
        # Skip alpaca_eval.yaml since it's already been run
        # if [[ "$config_file" == *"alpaca_eval.yaml" ]]; then
        #     echo "Skipping: $config_file (already processed)"
        #     continue
        # fi

        echo "Processing: $config_file"
        python "$SCRIPT" --config_path "$config_file"
        echo "Completed: $config_file"
        echo "---"
    fi
done

echo "All configurations processed."