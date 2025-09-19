#!/usr/bin/env python3
"""
Combine learned subspaces and calculated steering matrices into a single file
for use with our subspace steering model.

This script takes:
1. Learned subspaces from learn_benign_subspaces.py (W, m, V matrices)
2. Calculated steering matrices from calc_subspace_steering_matrix.py (Delta matrices)

And combines them into the format expected by SubspaceAlphaLlamaForCausalLM.
"""

import os
import pickle
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Combine subspace data for steering")
    parser.add_argument("--learned_subspaces_path", type=str, required=True,
                       help="Path to learned subspaces pickle file")
    parser.add_argument("--steering_matrices_path", type=str, required=True,
                       help="Path to calculated steering matrices pickle file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for combined subspace data")
    parser.add_argument("--model_name", type=str, default="llama3.1",
                       help="Model name for logging")
    return parser.parse_args()

def load_learned_subspaces(path):
    """Load learned subspaces from learn_benign_subspaces.py output"""
    logger.info(f"Loading learned subspaces from: {path}")

    with open(path, 'rb') as f:
        learned_subspaces = pickle.load(f)

    logger.info(f"Loaded subspaces for layers: {list(learned_subspaces.keys())}")

    # Log details for each layer
    for layer_idx, layer_data in learned_subspaces.items():
        num_subspaces = len(layer_data['W'])
        logger.info(f"Layer {layer_idx}: {num_subspaces} subspaces")
        logger.info(f"  - W matrices: {[W.shape for W in layer_data['W']]}")
        logger.info(f"  - m centers: {[m.shape for m in layer_data['m']]}")
        logger.info(f"  - V residual: {[V.shape for V in layer_data['V']]}")

    return learned_subspaces

def load_steering_matrices(path):
    """Load calculated steering matrices from calc_subspace_steering_matrix.py output"""
    logger.info(f"Loading steering matrices from: {path}")

    with open(path, 'rb') as f:
        steering_matrices = pickle.load(f)

    logger.info(f"Loaded steering matrices for layers: {list(steering_matrices.keys())}")

    # Log details for each layer
    for layer_idx, layer_data in steering_matrices.items():
        num_deltas = len(layer_data['Delta'])
        logger.info(f"Layer {layer_idx}: {num_deltas} Delta matrices")
        logger.info(f"  - Delta shapes: {[Delta.shape for Delta in layer_data['Delta']]}")

    return steering_matrices

def combine_subspace_data(learned_subspaces, steering_matrices):
    """
    Combine learned subspaces and steering matrices into format expected by our model.

    Expected format:
    {
        layer_idx: {
            'W': [W_0, W_1, ...],      # Original OT-learned subspace bases
            'm': [m_0, m_1, ...],      # Subspace centers/origins
            'V': [V_0, V_1, ...],      # Residual spaces (for steering)
            'Delta': [Delta_0, Delta_1, ...] # Learned steering matrices
        }
    }
    """
    logger.info("Combining subspace data...")

    combined_data = {}

    # Get all layers that have both learned subspaces and steering matrices
    common_layers = set(learned_subspaces.keys()) & set(steering_matrices.keys())
    logger.info(f"Layers with both subspaces and steering matrices: {sorted(common_layers)}")

    for layer_idx in sorted(common_layers):
        logger.info(f"Processing layer {layer_idx}...")

        learned_layer = learned_subspaces[layer_idx]
        steering_layer = steering_matrices[layer_idx]

        # Verify that the number of subspaces matches
        num_learned = len(learned_layer['W'])
        num_steering = len(steering_layer['Delta'])

        if num_learned != num_steering:
            logger.warning(f"Layer {layer_idx}: Mismatch in subspace count - "
                          f"learned: {num_learned}, steering: {num_steering}")
            # Use the minimum to avoid index errors
            num_subspaces = min(num_learned, num_steering)
            logger.warning(f"Using {num_subspaces} subspaces for this layer")
        else:
            num_subspaces = num_learned
            logger.info(f"Layer {layer_idx}: {num_subspaces} subspaces")

        # Combine the data
        combined_data[layer_idx] = {
            'W': learned_layer['W'][:num_subspaces],      # Original subspace bases
            'm': learned_layer['m'][:num_subspaces],      # Subspace centers
            'V': learned_layer['V'][:num_subspaces],      # Residual spaces
            'Delta': steering_layer['Delta'][:num_subspaces]  # Steering matrices
        }

        logger.info(f"Layer {layer_idx}: Combined {num_subspaces} subspaces successfully")

    logger.info(f"Combined data for {len(combined_data)} layers: {sorted(combined_data.keys())}")
    return combined_data

def save_combined_data(combined_data, output_path):
    """Save combined subspace data to pickle file"""
    logger.info(f"Saving combined data to: {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)

    logger.info(f"Successfully saved combined subspace data")

    # Log file size
    file_size = os.path.getsize(output_path) / (1024**2)  # MB
    logger.info(f"Output file size: {file_size:.2f} MB")

def main():
    args = parse_args()

    logger.info(f"=== Combining Subspace Data for {args.model_name} ===")

    # Validate input files exist
    if not os.path.exists(args.learned_subspaces_path):
        raise FileNotFoundError(f"Learned subspaces file not found: {args.learned_subspaces_path}")

    if not os.path.exists(args.steering_matrices_path):
        raise FileNotFoundError(f"Steering matrices file not found: {args.steering_matrices_path}")

    try:
        # Load the input data
        learned_subspaces = load_learned_subspaces(args.learned_subspaces_path)
        steering_matrices = load_steering_matrices(args.steering_matrices_path)

        # Combine the data
        combined_data = combine_subspace_data(learned_subspaces, steering_matrices)

        # Save the result
        save_combined_data(combined_data, args.output_path)

        logger.info("=== Subspace Data Combination Complete ===")
        logger.info(f"Combined data available at: {args.output_path}")
        logger.info("Ready for use with SubspaceAlphaLlamaForCausalLM!")

    except Exception as e:
        logger.error(f"Error during combination: {e}")
        raise

if __name__ == "__main__":
    main()