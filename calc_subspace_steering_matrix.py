import torch
torch.manual_seed(42)
import sys
import os
sys.path.append('src')
from utils.const import AlphaSteer_STEERING_LAYERS
from get_subspace_steering_matrix import SafetyEnhancedSteering
from learn_subspace import OTSubspaceLearner

import pickle
import torch
import os
import argparse
import numpy as np

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate subspace-specific steering matrices")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3.1, qwen2.5, gemma2)")
    parser.add_argument("--subspaces_path", type=str, required=True, help="Path to learned subspaces file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save steering matrices")

    # Delta learning parameters
    parser.add_argument("--gamma", type=float, default=1e-3, help="Regularization weight for delta learning")

    return parser.parse_args()

def load_benign_embeddings(embedding_dir, device):
    """Load and combine benign embeddings same as calc_steering_matrix.py"""
    logger.info("Loading benign embeddings...")

    # Load benign embeddings
    H_benign_train_10000 = torch.load(f"{embedding_dir}/embeds_benign_train.pt", map_location=device).float()
    H_coconot_pref = torch.load(f"{embedding_dir}/embeds_coconot_pref.pt", map_location=device).float()
    H_coconot_original = torch.load(f"{embedding_dir}/embeds_coconot_original.pt", map_location=device).float()

    # Sample a subset of borderline examples to balance the dataset
    indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]

    # Combine all benign embeddings
    H_benign_train = torch.cat([
        H_benign_train_10000, H_coconot_original[indices_borderline], H_coconot_pref], dim=0).to(device)

    logger.info(f"H_benign_train shape: {H_benign_train.shape}")
    return H_benign_train

def load_harmful_embeddings(embedding_dir, device):
    """Load and combine harmful embeddings same as calc_steering_matrix.py"""
    logger.info("Loading harmful embeddings...")

    # Load harmful embeddings
    H_harmful_train_1000 = torch.load(f"{embedding_dir}/embeds_harmful_train_1000.pt", map_location=device).float()
    H_jailbreak_train_full = torch.load(f"{embedding_dir}/embeds_jailbreak_train.pt", map_location=device).float()

    # Sample a subset of jailbreak examples
    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]

    # Combine all harmful embeddings
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)

    logger.info(f"H_harmful_train shape: {H_harmful_train.shape}")
    return H_harmful_train

def load_learned_subspaces(subspaces_path, device):
    """Load previously learned subspaces"""
    logger.info(f"Loading learned subspaces from {subspaces_path}")

    with open(subspaces_path, 'rb') as f:
        learned_subspaces = pickle.load(f)

    # Move to device
    for layer in learned_subspaces:
        for key in ['U', 'V', 'm']:
            if key in learned_subspaces[layer]:
                learned_subspaces[layer][key] = [
                    tensor.to(device) for tensor in learned_subspaces[layer][key]
                ]

    logger.info(f"Loaded subspaces for layers: {list(learned_subspaces.keys())}")
    return learned_subspaces

def assign_data_to_subspaces(H_benign, H_harmful, learned_subspaces, target_layers, device):
    """Assign benign and harmful data to closest subspaces for each layer"""
    layer_assignments = {}

    for layer in target_layers:
        if layer not in learned_subspaces:
            logger.warning(f"Layer {layer} not found in learned subspaces, skipping")
            continue

        logger.info(f"Assigning data to subspaces for layer {layer}")

        # Extract layer data
        H_benign_layer = H_benign[:, layer, :]  # (N_benign, d)
        H_harmful_layer = H_harmful[:, layer, :] # (N_harmful, d)

        # Get subspace components for this layer
        U_layer = learned_subspaces[layer]['U']
        V_layer = learned_subspaces[layer]['V']
        m_layer = learned_subspaces[layer]['m']
        K = len(V_layer)

        # Create steering class to use closest_subspace method
        steering = SafetyEnhancedSteering(U_layer, V_layer, m_layer, device)

        # Assign benign data to subspaces
        benign_assignments = []
        for i in range(H_benign_layer.shape[0]):
            k = steering.closest_subspace(H_benign_layer[i])
            benign_assignments.append(k)
        benign_assignments = torch.tensor(benign_assignments)

        # Assign harmful data to subspaces
        harmful_assignments = []
        for i in range(H_harmful_layer.shape[0]):
            k = steering.closest_subspace(H_harmful_layer[i])
            harmful_assignments.append(k)
        harmful_assignments = torch.tensor(harmful_assignments)

        # Group data by subspace assignment
        D_b = []  # benign data per subspace
        D_m = []  # harmful data per subspace

        for k in range(K):
            # Get benign data assigned to subspace k
            benign_mask = (benign_assignments == k)
            D_b_k = H_benign_layer[benign_mask]  # (N_bk, d)
            D_b.append(D_b_k)

            # Get harmful data assigned to subspace k
            harmful_mask = (harmful_assignments == k)
            D_m_k = H_harmful_layer[harmful_mask]  # (N_mk, d)
            D_m.append(D_m_k)

            logger.info(f"Layer {layer}, Subspace {k}: {D_b_k.shape[0]} benign, {D_m_k.shape[0]} harmful")

        layer_assignments[layer] = {
            'steering': steering,
            'D_b': D_b,
            'D_m': D_m,
            'benign_assignments': benign_assignments,
            'harmful_assignments': harmful_assignments
        }

        # Clean up
        del steering, H_benign_layer, H_harmful_layer
        torch.cuda.empty_cache()

    return layer_assignments

def learn_steering_matrices(layer_assignments, args):
    """Learn delta matrices for each subspace in each layer"""
    logger.info("Learning steering matrices...")

    steering_matrices = {}

    for layer, assignment_data in layer_assignments.items():
        logger.info(f"Learning steering matrices for layer {layer}")

        steering = assignment_data['steering']
        D_b = assignment_data['D_b']
        D_m = assignment_data['D_m']

        # Learn delta matrices using the steering class
        steering.learn_delta(D_b, D_m, gamma=args.gamma)

        # Store the learned delta matrices
        steering_matrices[layer] = {
            'U': [U.cpu() for U in steering.U],  # Principal subspaces
            'V': [V.cpu() for V in steering.V],  # Residual subspaces
            'm': [m.cpu() for m in steering.m],  # Subspace origins
            'Delta': [Delta.cpu() for Delta in steering.Delta],  # Learned steering matrices
            'gamma': args.gamma
        }

        logger.info(f"Layer {layer}: Learned {len(steering.Delta)} steering matrices")

    return steering_matrices

if __name__ == "__main__":
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Get target layers
    target_layers = AlphaSteer_STEERING_LAYERS[args.model_name]
    logger.info(f"Target layers: {target_layers}")

    # Load embeddings
    H_benign_train = load_benign_embeddings(args.embedding_dir, device)
    H_harmful_train = load_harmful_embeddings(args.embedding_dir, device)

    # Load learned subspaces
    learned_subspaces = load_learned_subspaces(args.subspaces_path, device)

    # Assign data to subspaces
    layer_assignments = assign_data_to_subspaces(
        H_benign_train, H_harmful_train, learned_subspaces, target_layers, device
    )

    # Learn steering matrices
    steering_matrices = learn_steering_matrices(layer_assignments, args)

    # Save steering matrices
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f"{args.save_dir}/steering_matrices_{args.model_name}.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(steering_matrices, f)

    logger.info(f"Steering matrices saved to {save_path}")

    # Summary
    logger.info("=== Steering Matrix Learning Summary ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layers processed: {list(steering_matrices.keys())}")
    logger.info(f"Regularization gamma: {args.gamma}")
    for layer in steering_matrices:
        K = len(steering_matrices[layer]['Delta'])
        logger.info(f"Layer {layer}: {K} subspace-specific steering matrices learned")