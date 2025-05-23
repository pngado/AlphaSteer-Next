import os
import logging
import datetime

import torch.nn as nn
import torch
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Add these imports
from typing import Optional, Tuple, Union, List, Dict, Unpack
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import FlashAttentionKwargs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Only CausalLM can be called from outside
__all__ = [
    'AlphaLlamaForCausalLM',
    ]


class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, 
                 layer_idx: int, 
                 steering_matrix: Optional[torch.Tensor] = None, 
                 strength: float = 0.0
                 ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        # self.steering_vector = None
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            self.steering_matrix = steering_matrix.to(device)
        else:
            self.steering_matrix = None
        self.strength = strength
        
    def set_steering_parameters(
        self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: float = 0.0,
        device: Optional[torch.device]=None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength
        # self.steering_vector = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Ensure steering_matrix is on the same device as hidden_states
        if hidden_states.shape[1] > 1: # Only apply steering on initial input
            if self.steering_matrix is not None and torch.any(self.steering_matrix):
                # Only apply steering once during input processing
                if self.steering_matrix.device != hidden_states.device:
                    self.steering_matrix = self.steering_matrix.to(hidden_states.device)
                # Calculate steering vector by multiplying the last token's hidden state with the steering matrix
                steering_vector = hidden_states[:, -1, :] @ self.steering_matrix * self.strength
                # Reshape to match hidden_states dimensions and move to the same device
                steering_vector = steering_vector.unsqueeze(1).to(hidden_states.device)
                # Apply steering by adding the steering vector to hidden states
                hidden_states = hidden_states + steering_vector
                
                # self.steering_vector = hidden_states[:, -1, :] @ self.steering_matrix * self.strength
                # self.steering_vector = self.steering_vector.unsqueeze(1).to(hidden_states.device) # Same dimensions as hidden_states
        # if self.steering_vector is not None:
        #     if self.steering_vector.device != hidden_states.device:
        #         self.steering_vector = self.steering_vector.to(hidden_states.device)
            
        #     hidden_states = hidden_states + self.steering_vector
            
        residual = hidden_states # resid_pre - save for residual connection

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
        residual = hidden_states # resid_mid - save after attention residual
        
        # Normalize hidden states after attention, then pass through MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # resid_post - final residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


    def fnn_output(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Forward pass through the layer, but stop at the MLP output (before residual connection).
        """
        residual = hidden_states # resid pre
        
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.self_attn(
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
        residual = hidden_states  # Update residual
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)

        return mlp_output

class AlphaLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Replace the layers with our custom Alpha layers, keeping everything else unchanged
        self.layers = nn.ModuleList(
            [AlphaLlamaDecoderLayer(
                config=config, 
                layer_idx=layer_idx,
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: Optional[list[float]] = None,
        device: Optional[torch.device] = None):
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx]
                
            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix, 
                strength=strength[layer_idx] if strength is not None else 0.0
            )
            torch.cuda.empty_cache()
        
        self.print_steering_parameters()
        
    def print_steering_parameters(self):
        logger.info("Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'Steering Matrix (First Element)'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            # Ensure strength is a string or formattable type
            strength_val = str(layer.strength)
            
            if layer.steering_matrix is not None:
                steering_matrix_str = layer.steering_matrix[0, 0]
            else:
                steering_matrix_str = "None"
            logger.info(f"{layer_idx:<10}{strength_val:<20}{steering_matrix_str}")
    


class AlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = AlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, 
                        steering_matrix: Optional[torch.Tensor] = None,
                        strength: Optional[list[float]] = None,
                        **kwargs):
        # Call the parent class's from_pretrained method to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return model

    def set_steering_parameters(
            self, 
            steering_matrix: Optional[torch.Tensor]=None, 
            strength: Optional[list[float]] = None):
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        self.model.set_steering_parameters(
            steering_matrix=steering_matrix, 
            strength=strength,
            device=device
        )
        
    def analyze_embed_vector(
        self,
        embed_vector: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
        device: Optional[torch.device] = None,
    ) -> List[Tuple[int, float]]:
        """
        Analyze an embedding vector to find the top-k tokens it's most likely to generate.
        
        Args:
            embed_vector: The embedding vector to analyze
            layer_idx: Which layer to analyze
            top_k: Number of top tokens to return
            device: Device to run computation on
            
        Returns:
            List of (token_id, score) tuples for the top-k tokens
        """
        device = device or next(self.parameters()).device
        embed_vector = embed_vector.to(device)  # Shape: (d_model,)

        # Ensure the embedding vector has the correct shape
        if embed_vector.dim() != 1:
            raise ValueError(f"embed_vector must be 1D, but got {embed_vector.dim()}D.")

        # Add batch and sequence dimensions to match decoder input format
        hidden_states = embed_vector.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_model)

        target_layer = self.model.layers[layer_idx]
        attention_mask = None  # No masking required for single token
        position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)  # Single position (0)
        past_key_values = None  # No past key values for this analysis
        
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        hidden_states = target_layer.fnn_output(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )  # Shape: (1, 1, d_model)
        
        final_hidden_state = hidden_states.squeeze(0).squeeze(0)  # Shape: (d_model,)

        if hasattr(self.lm_head, "weight"):
            embedding_matrix = self.lm_head.weight  # Shape: (vocab_size, d_model)
        else:
            raise ValueError("Could not find the output embedding matrix in `lm_head`.")

        vocab_scores = torch.matmul(embedding_matrix, final_hidden_state)  # Shape: (vocab_size,)
        top_k_scores, top_k_indices = torch.topk(vocab_scores, top_k)
        top_tokens = [(token_id.item(), score.item()) for token_id, score in zip(top_k_indices, top_k_scores)]

        return top_tokens