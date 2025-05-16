import torch
torch.manual_seed(42)
from .utils.const import NSS_CALCULATION_CONFIG
from .utils.steering_utils import *

import pickle
import torch
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "data"
save_dir = f"{base_dir}/steering_matrix"

if __name__ == "__main__":
    for model_name in NSS_CALCULATION_CONFIG.keys():
        
        logger.info(f"model_name: {model_name}")
        embeds_dir = f"{base_dir}/embeddings/{model_name}"
        # Get layer-specific configuration (layer number and nullspace ratio)
        layers_ratio_list = NSS_CALCULATION_CONFIG[model_name]
        
        # Load benign embeddings
        H_benign_train_10000 = torch.load(f"{embeds_dir}/embeds_benign_train.pt", map_location=device)
        H_coconot_pref = torch.load(f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device)
        H_coconot_original = torch.load(f"{embeds_dir}/embeds_coconot_original.pt", map_location=device)
        
        H_math_train = torch.load(f"{embeds_dir}/embeds_math_train.pt", map_location=device)
        H_gsm8k_train = torch.load(f"{embeds_dir}/embeds_gsm8k_train.pt", map_location=device)

        # Sample a subset of borderline examples to balance the dataset
        indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
        # Combine all benign embeddings
        H_benign_train = torch.cat([
            H_benign_train_10000, H_coconot_original[indices_borderline], H_coconot_pref], dim=0).to(device)
        
        logger.info(f"H_benign_train shape: {H_benign_train.shape}")
        torch.cuda.empty_cache()
        
        # Load harmful embeddings
        H_harmful_train_1000 = torch.load(f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location=device)
        H_jailbreak_train_full = torch.load(f"{embeds_dir}/embeds_jailbreak_train.pt", map_location=device)
        
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
        refusal_vectors_path = f"{base_dir}/refusal_vectors/{model_name}/refusal.pkl"
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
        save_path = os.path.join(save_dir, model_name,f"steering_matrix_with_regularization_v2_lambda_10_1.pt")
        torch.save(steering_matrix, save_path)
        logger.info(f"steering matrix saved to {save_path}")

        # Clean up memory
        H_benign_train = None; H_harmful_train = None
        P = None; tilde_delta = None; steering_matrix = None
        torch.cuda.empty_cache()