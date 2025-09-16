import torch
torch.manual_seed(42)
import sys
import os
sys.path.append('src')
from utils.const import AlphaSteer_STEERING_LAYERS
from learn_subspace import OTSubspaceLearner

import pickle
import torch
import os
import argparse

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Learn subspaces from benign embeddings")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3.1, qwen2.5, gemma2)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save learned subspaces")

    # Subspace learning parameters
    parser.add_argument("--K", type=int, default=16, help="Number of subspaces to learn")
    parser.add_argument("--p_ratio", type=float, default=0.25, help="Subspace dimension as ratio of embedding dim")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="OT regularization parameter")
    parser.add_argument("--alpha", type=float, default=0.1, help="Variance preservation ratio (1-alpha kept as principal)")

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

def learn_subspaces_per_layer(H_benign, target_layers, args, device):
    """Learn subspaces for each specified layer"""
    num_samples, num_layers, d_model = H_benign.shape
    p = int(d_model * args.p_ratio)  # subspace dimension

    logger.info(f"Learning subspaces with K={args.K}, p={p}, d={d_model}")
    logger.info(f"Alpha={args.alpha} (keeping {1-args.alpha:.1%} variance as principal)")

    learned_subspaces = {}

    for layer in target_layers:
        logger.info(f"Learning subspaces for layer {layer}")

        # Extract embeddings for this layer
        X_layer = H_benign[:, layer, :]  # (num_samples, d_model)

        # Initialize and train subspace learner
        learner = OTSubspaceLearner(
            d=d_model,
            p=p,
            K=args.K,
            epsilon=args.epsilon,
            lr=args.lr,
            device=device
        )

        # Train subspaces
        logger.info(f"Training subspaces for layer {layer}...")
        learner.fit(X_layer, epochs=args.epochs)

        # Refine subspaces to get principal and residual components
        logger.info(f"Refining subspaces for layer {layer}...")
        U_list, V_list = learner.refine_subspaces(X_layer, alpha=args.alpha)

        # Store learned parameters
        learned_subspaces[layer] = {
            'W': [W.detach().cpu() for W in learner.W],  # initial subspace bases
            'm': [m.detach().cpu() for m in learner.m],  # subspace means/origins
            'U': [U.detach().cpu() for U in U_list],     # principal subspaces (90% variance)
            'V': [V.detach().cpu() for V in V_list],     # residual subspaces (for steering)
            'alpha': args.alpha                          # variance preservation parameter
        }

        logger.info(f"Layer {layer}: Learned {args.K} subspaces")
        logger.info(f"Principal dims: {[U.shape[1] if U.shape[1] < d_model else 'full' for U in U_list]}")
        logger.info(f"Residual dims: {[V.shape[1] if V.shape[1] > 0 else 'none' for V in V_list]}")

        # Clean up GPU memory
        del learner, X_layer
        torch.cuda.empty_cache()

    return learned_subspaces

if __name__ == "__main__":
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Get target layers for this model
    target_layers = AlphaSteer_STEERING_LAYERS[args.model_name]
    logger.info(f"Target layers: {target_layers}")

    # Load benign embeddings
    H_benign_train = load_benign_embeddings(args.embedding_dir, device)

    # Learn subspaces for each layer
    learned_subspaces = learn_subspaces_per_layer(
        H_benign_train, target_layers, args, device
    )

    # Save learned subspaces
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f"{args.save_dir}/learned_subspaces_{args.model_name}.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(learned_subspaces, f)

    logger.info(f"Learned subspaces saved to {save_path}")

    # Summary
    logger.info("=== Subspace Learning Summary ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layers processed: {list(learned_subspaces.keys())}")
    logger.info(f"Subspaces per layer: {args.K}")
    logger.info(f"Subspace dimension: {int(H_benign_train.shape[2] * args.p_ratio)}")
    logger.info(f"Total samples used: {H_benign_train.shape[0]}")
    logger.info(f"Principal variance preserved: {1-args.alpha:.1%}")