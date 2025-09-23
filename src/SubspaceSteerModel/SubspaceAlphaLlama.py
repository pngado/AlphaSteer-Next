import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

# Import AlphaSteer's proven architecture
from AlphaSteerModel.AlphaLlama import (
    AlphaLlamaDecoderLayer,
    AlphaLlamaModel,
    AlphaLlamaForCausalLM
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Only export the CausalLM model
__all__ = [
    'SubspaceAlphaLlamaForCausalLM',
]


class SubspaceAlphaLlamaDecoderLayer(AlphaLlamaDecoderLayer):
    """
    Custom decoder layer that applies subspace-specific steering instead of global steering.
    Inherits from AlphaSteer's proven architecture.
    """

    def __init__(self, config, layer_idx, subspace_data=None, strength=0.0):
        # Initialize with AlphaSteer's proven architecture (but no steering matrix)
        super().__init__(config, layer_idx, steering_matrix=None, strength=0.0)

        # Replace AlphaSteer's single steering matrix with subspace data
        if subspace_data:
            self.subspace_W = subspace_data['W']        # List of W_k subspace bases
            self.subspace_m = subspace_data['m']        # List of m_k subspace centers
            self.subspace_V = subspace_data['V']        # List of V_k residual spaces
            self.subspace_Delta = subspace_data['Delta'] # List of Delta_k steering matrices
            self.valid_subspaces = subspace_data.get('valid_subspaces', list(range(len(subspace_data['W']))))  # NEW
            self.num_subspaces = len(self.subspace_W)
        else:
            self.subspace_W = None
            self.subspace_m = None
            self.subspace_V = None
            self.subspace_Delta = None
            self.valid_subspaces = []
            self.num_subspaces = 0

        self.subspace_strength = strength

    def set_subspace_parameters(self, subspace_data=None, strength=0.0, device=None):
        """Set subspace steering parameters for this layer"""
        device = next(self.parameters()).device if device is None else device

        if subspace_data:
            # Move all subspace data to device
            self.subspace_W = [W.to(device) for W in subspace_data['W']]
            self.subspace_m = [m.to(device) for m in subspace_data['m']]
            self.subspace_V = [V.to(device) for V in subspace_data['V']]
            self.subspace_Delta = [Delta.to(device) for Delta in subspace_data['Delta']]
            self.valid_subspaces = subspace_data.get('valid_subspaces', list(range(len(subspace_data['W']))))  # NEW
            self.num_subspaces = len(self.subspace_W)

        self.subspace_strength = strength

    def find_closest_subspace(self, h: torch.Tensor) -> int:
        """
        Find closest subspace using W_k @ W_k.T @ (h - m_k) projection.
        Only considers subspaces that had harmful training data.
        Returns the index of the closest valid subspace.
        """
        if self.num_subspaces == 0 or len(self.valid_subspaces) == 0:
            return 0

        min_distance = float('inf')
        closest_idx = self.valid_subspaces[0]  # Default to first valid subspace

        # Only check valid subspaces (those with harmful training data)
        for k in self.valid_subspaces:
            W_k = self.subspace_W[k]  # [d_model, p_k]
            m_k = self.subspace_m[k]  # [d_model]

            # Center the point
            h_centered = h - m_k  # [d_model]

            # Project to subspace k: W_k @ W_k.T @ (h - m_k)
            projection = W_k @ (W_k.T @ h_centered)  # [d_model]

            # Calculate residual distance
            residual = h_centered - projection  # [d_model]
            distance = torch.norm(residual).item()

            if distance < min_distance:
                min_distance = distance
                closest_idx = k

        return closest_idx

    def apply_subspace_steering(self, h: torch.Tensor, subspace_idx: int) -> torch.Tensor:
        """
        Apply steering in the V_k residual space using Delta_k matrix.
        Args:
            h: hidden state vector [d_model]
            subspace_idx: which subspace to use for steering
        Returns:
            steering_vector: [d_model]
        """
        if subspace_idx >= self.num_subspaces or subspace_idx not in self.valid_subspaces:
            return torch.zeros_like(h)

        V_k = self.subspace_V[subspace_idx]      # [d_model, residual_dim]
        m_k = self.subspace_m[subspace_idx]      # [d_model]
        Delta_k = self.subspace_Delta[subspace_idx]  # [residual_dim, d_model]

        # Center around subspace origin
        h_centered = h - m_k  # [d_model]

        # Project to residual space V_k
        residual_proj = h_centered @ V_k @ V_k.T  # [d_model]

        # Apply steering using Delta_k matrix
        steering_vector = residual_proj @ Delta_k.T * self.subspace_strength  # [d_model]

        return steering_vector

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Handle tuple input (extract tensor from tuple if needed)
        # if isinstance(hidden_states, tuple):
        #     hidden_states = hidden_states[0]

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                f"Expected hidden_states to be a torch.Tensor or nested tuple/list "
                f"containing one, but got type {type(hidden_states)}"
            )

        # Apply subspace steering (replaces AlphaSteer's global steering)
        if hidden_states.shape[1] > 1:  # Only apply steering on initial input
            if self.subspace_W is not None and self.subspace_strength > 0.0:
                batch_size = hidden_states.shape[0]
                last_token = hidden_states[:, -1, :]  # [batch_size, d_model]

                # Apply subspace steering to each sample in batch
                for batch_idx in range(batch_size):
                    h = last_token[batch_idx]  # [d_model]

                    # Step 1: Find closest subspace using W_k projection
                    subspace_idx = self.find_closest_subspace(h)

                    # Step 2: Apply subspace-specific steering
                    steering_vector = self.apply_subspace_steering(h, subspace_idx)

                    # Step 3: Apply steering to all positions for this sample
                    steering_vector = steering_vector.unsqueeze(0)  # [1, d_model]
                    hidden_states[batch_idx] = hidden_states[batch_idx] + steering_vector

        # Continue with AlphaSteer's proven transformer processing
        residual = hidden_states  # resid_pre - save for residual connection

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states  # resid_mid - save after attention residual

        # Normalize hidden states after attention, then pass through MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # resid_post - final residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

        # if output_attentions:
        #     return (hidden_states, self_attn_weights)
        # return hidden_states


class SubspaceAlphaLlamaModel(AlphaLlamaModel):
    """
    Model that uses subspace steering layers instead of global steering layers.
    Inherits from AlphaSteer's proven model architecture.
    """

    def __init__(self, config):
        # Initialize with AlphaSteer's architecture but replace layers
        super().__init__(config)

        # Replace AlphaSteer layers with our subspace steering layers
        self.layers = nn.ModuleList(
            [SubspaceAlphaLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_subspace_steering_parameters(
        self,
        subspace_data_dict: Optional[Dict[int, Dict]] = None,
        strength: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Set subspace steering parameters for all layers.

        Args:
            subspace_data_dict: {layer_idx: {'W': [W_0, W_1, ...], 'm': [m_0, m_1, ...],
                                           'V': [V_0, V_1, ...], 'Delta': [Delta_0, Delta_1, ...]}}
            strength: List of strength values, one per layer
            device: Device to move tensors to
        """
        device = next(self.parameters()).device if device is None else device

        for layer_idx, layer in enumerate(self.layers):
            layer_subspace_data = None
            layer_strength = 0.0

            # Get subspace data for this layer if available
            if subspace_data_dict and layer_idx in subspace_data_dict:
                layer_subspace_data = subspace_data_dict[layer_idx]

            # Get strength for this layer
            if strength and layer_idx < len(strength):
                layer_strength = strength[layer_idx]

            # Set parameters for this layer
            layer.set_subspace_parameters(
                subspace_data=layer_subspace_data,
                strength=layer_strength,
                device=device
            )

            # Clean up GPU memory
            torch.cuda.empty_cache()

        self.print_subspace_parameters()

    def print_subspace_parameters(self):
        """Print subspace steering parameters for debugging"""
        logger.info("Subspace Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<15}{'Num Subspaces':<15}{'Has Data'}")
        logger.info("=" * 60)

        for layer_idx, layer in enumerate(self.layers):
            strength_val = f"{layer.subspace_strength:.3f}"
            num_subspaces = str(layer.num_subspaces)
            has_data = "Yes" if layer.subspace_W is not None else "No"

            logger.info(f"{layer_idx:<10}{strength_val:<15}{num_subspaces:<15}{has_data}")


class SubspaceAlphaLlamaForCausalLM(AlphaLlamaForCausalLM):
    """
    Top-level model class for subspace steering.
    Inherits from AlphaSteer's proven CausalLM architecture.
    """

    def __init__(self, config):
        # Initialize with AlphaSteer's architecture but use our custom model
        super().__init__(config)
        self.model = SubspaceAlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        subspace_data: Optional[Dict] = None,
        strength: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Load pretrained model and optionally set subspace steering parameters.

        Args:
            pretrained_model_name_or_path: Model path or HuggingFace model name
            subspace_data: Dictionary with subspace data for each layer
            strength: List of strength values for each layer
        """
        # Call parent class to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Set subspace steering parameters if provided
        if subspace_data is not None:
            model.set_subspace_steering_parameters(
                subspace_data_dict=subspace_data,
                strength=strength
            )

        return model

    def set_subspace_steering_parameters(
        self,
        subspace_data_dict: Optional[Dict] = None,
        strength: Optional[List[float]] = None
    ):
        """Set subspace steering parameters for the model"""
        device = next(self.parameters()).device

        self.model.set_subspace_steering_parameters(
            subspace_data_dict=subspace_data_dict,
            strength=strength,
            device=device
        )