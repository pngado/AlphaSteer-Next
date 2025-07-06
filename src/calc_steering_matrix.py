import torch
torch.manual_seed(42)
from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.steering_utils import *

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
    parser = argparse.ArgumentParser(description="Calculate steering matrix for AlphaSteer")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3.1, qwen2.5, gemma2)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the steering matrix")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    # Set device
    device = torch.device(args.device)

    logger.info(f"model_name: {args.model_name}")
    embeds_dir = args.embedding_dir
    # Get layer-specific configuration (layer number and nullspace ratio)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    
    # Load benign embeddings
    H_benign_train_10000 = torch.load(f"{embeds_dir}/embeds_benign_train.pt", map_location=device).float()
    H_coconot_pref = torch.load(f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device).float()
    H_coconot_original = torch.load(f"{embeds_dir}/embeds_coconot_original.pt", map_location=device).float()
    
    # Sample a subset of borderline examples to balance the dataset
    indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
    # Combine all benign embeddings
    H_benign_train = torch.cat([
        H_benign_train_10000, H_coconot_original[indices_borderline], H_coconot_pref], dim=0).to(device)
    
    logger.info(f"H_benign_train shape: {H_benign_train.shape}")
    torch.cuda.empty_cache()
    
    # Load harmful embeddings
    H_harmful_train_1000 = torch.load(f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location=device).float()
    H_jailbreak_train_full = torch.load(f"{embeds_dir}/embeds_jailbreak_train.pt", map_location=device).float()
    
    # Sample a subset of jailbreak examples
    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    # Combine all harmful embeddings
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)
    # Free memory
    H_harmful_train_1000 = None; H_jailbreak_train_full = None; H_jailbreak_additional = None
    torch.cuda.empty_cache()
    logger.info(f"H_harmful_train.shape: {H_harmful_train.shape}")
    
    # Load refusal vectors (directions that lead to refusal responses)
    refusal_vectors_path = f"data/refusal_vectors/RV/{args.model_name}_RV_refusal.pkl"
    refusal_vectors = pickle.load(open(refusal_vectors_path, "rb"))
    refusal_vectors = torch.tensor(
        refusal_vectors, dtype=torch.float32).to(device)
    logger.info("refusal vectors' shape: %s", refusal_vectors.shape)

    logger.info(f"H_benign_train.shape: {H_benign_train.shape}")

    # Initialize tensors to store projection matrices, delta matrices, and steering matrices
    num_layer = refusal_vectors.shape[0]
    d_model = refusal_vectors.shape[1]
    P = torch.zeros(num_layer, d_model, d_model, device=device)
    tilde_delta = torch.zeros(num_layer, d_model, d_model, device=device)
    steering_matrix = torch.zeros(num_layer, d_model, d_model, device=device)
    
    
    for layer, ratio in layers_ratio_list:
        logger.info(f"layer: {layer}, ratio: {ratio}")
        
        # Calculate null space projection matrix for benign embeddings
        P_layer = null_space_projection_l(H_benign_train[:, layer, :], abs_nullspace_ratio=ratio)
        P[layer] = P_layer
        P_norm = torch.norm(P_layer)
        logger.info(f"P_norm: {P_norm}")

        # Calculate delta matrix with regularization to align with refusal vectors
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_harmful_train[:, layer, :], P_layer, refusal_vectors[layer], lambda_reg=10.0, device=device)
        
        tilde_delta[layer] = tilde_delta_layer
        tilde_delta_norm = torch.norm(tilde_delta_layer)
        logger.info(f"tilde_delta_norm: {tilde_delta_norm}")

        # Calculate final steering matrix by combining projection and delta matrices
        steering_matrix_layer = cal_steering_matrix_l(
            P_layer, tilde_delta_layer, device=device)
        steering_matrix[layer] = steering_matrix_layer

        steering_matrix_norm = torch.norm(steering_matrix_layer)
        logger.info(f"steering matrix layer {layer} norm: {steering_matrix_norm}")

    # Save the steering matrix
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(steering_matrix, args.save_path)
    logger.info(f"steering matrix saved to {args.save_path}")

    # Clean up memory
    H_benign_train = None; H_harmful_train = None
    P = None; tilde_delta = None; steering_matrix = None
    torch.cuda.empty_cache()