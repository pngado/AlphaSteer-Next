import logging

import torch.nn as nn
import torch
from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from typing import Optional, Tuple, Union, List, Dict#, Unpack
from transformers.cache_utils import Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

__all__ = [
    'SteerQwen2ForCausalLM',
    ]

class SteerQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, 
                 layer_idx: int, 
                 steering_vector: Optional[torch.Tensor] = None, 
                 strength: float = 1.0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        if steering_vector is not None:
            self.register_buffer("steering_vector", steering_vector)
        else:
            hidden_dim = config.hidden_size
            self.register_buffer("steering_vector", torch.empty(hidden_dim))
        self.register_buffer("strength", torch.tensor(strength, dtype=torch.float))

    def set_steering_parameters(
        self, 
        steering_vector: Optional[torch.Tensor]=None, 
        strength: float = 0.0,
        device: Optional[torch.device] = None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_vector is not None:
            self.steering_vector.data = steering_vector.to(device)

        self.strength.data = torch.tensor(strength, dtype=torch.float)
        
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
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Apply steering vector
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states + self.steering_vector * self.strength
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

class SteerQwen2Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, 
                 steering_vector: Optional[torch.Tensor] = None,
                 strength: Optional[List[float]] = None):
        super().__init__(config)
        self.steering_vector = steering_vector
        self.strength = strength if strength is not None else [0.0] * config.num_hidden_layers
        self.layers = nn.ModuleList(
            [SteerQwen2DecoderLayer(
                config=config, 
                layer_idx=layer_idx, 
                steering_vector=steering_vector[layer_idx] if steering_vector is not None else None, 
                strength=self.strength[layer_idx] if self.strength is not None else 0.0,
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self, 
        steering_vector: Optional[torch.Tensor] = None, 
        strength: Optional[list[float]] = None,
        device: Optional[torch.device] = None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_vector is not None:
            steering_vector = steering_vector.to(device)
            
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_vector = None
            if steering_vector is not None:
                layer_steering_vector = steering_vector[layer_idx]

            layer.set_steering_parameters(
                steering_vector=layer_steering_vector, 
                strength=strength[layer_idx] if strength is not None else 0.0
            )
            torch.cuda.empty_cache()
        self.print_steering_parameters()

    def print_steering_parameters(self):
        logger.info(f"{'Layer':<8}{'Strength':<12}{'Steering Vector'}")
        logger.info("-" * 32)
        for layer_idx, layer in enumerate(self.layers):
            strength = f"{layer.strength.item():.4f}"
            vector = "None" if layer.steering_vector is None or layer.steering_vector.nelement() == 0 else f"{layer.steering_vector[0].item():.4f}"
            logger.info(f"{layer_idx:<8}{strength:<12}{vector}")

class SteerQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config, 
                 steering_vector: Optional[torch.Tensor] = None,
                 strength: Optional[List[float]] = None):
        super().__init__(config)
        self.model = SteerQwen2Model(
            config=config, 
            steering_vector=steering_vector, 
            strength=strength
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, 
                        steering_vector: Optional[torch.Tensor] = None,
                        strength: Optional[list[float]] = None,
                        **kwargs):
        # Call the parent class's from_pretrained method to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_vector=steering_vector, strength=strength)
        return model

    def set_steering_parameters(
            self, 
            steering_vector: Optional[torch.Tensor] = None, 
            strength: Optional[list[float]] = None):
        
        # Ensure steering_matrix is on the device where the model is
        device = next(self.parameters()).device
        if steering_vector is not None:
            steering_vector = steering_vector.to(device)

        self.model.set_steering_parameters(
            steering_vector=steering_vector, 
            strength=strength,
            device=device
        )
