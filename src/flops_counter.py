import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pprint import pprint
from qwen import load_qwen
import sys

@dataclass
class ModelConfig: ### Just get the configeration of Qwen 2.5 from load_qwen() 
    """Configuration for Qwen2.5-Instruct model."""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    kv_channels: Optional[int] = None
    max_position_embeddings: int = 4096
    vocab_size: int = 151936
    group_query_attention: bool = True
    num_key_value_heads: Optional[int] = None  # For GQA
    seq_length: int = 512
    batch_size: int = 1

    def __post_init__(self):
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads if not self.group_query_attention else self.num_attention_heads // 8


class FLOPSCounter:
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def matrix_multiply_flops(self, m: int, n: int, p: int):
        # Each element requires n multiplications and n-1 additions
        flops_per_element = 2 * n - 1
        total_elements = m * p
        return total_elements * flops_per_element
    
    def layer_norm_flops(self, elements: int):
        # For each element: 
        # - Compute square: 1 multiplication
        # - Sum of squares: elements-1 additions
        # - Division by elements: 1 division
        # - Square root: 10 FLOPS
        # - Division by the root: 1 division
        # - Scaling: 1 multiplication
        
        flops_square = elements  # x²
        flops_sum = elements - 1  # sum(x²)
        flops_division = 1  # mean calculation
        flops_sqrt = 10  # sqrt operation
        flops_normalize = elements  # division by sqrt
        flops_scale = elements  # multiplication by scale
        
        return flops_square + flops_sum + flops_division + flops_sqrt + flops_normalize + flops_scale
    
    def swiglu_flops(self, elements: int):
        flops_sigmoid = elements * 12  # 1/(1+exp(-x))
        flops_swish = elements  # x * sigmoid(x)
        flops_multiply = elements  # elementwise multiplication
        
        return flops_sigmoid + flops_swish + flops_multiply
    
    def self_attention_flops(self):
        hidden_size = self.config.hidden_size
        seq_length = self.config.seq_length
        batch_size = self.config.batch_size
        num_heads = self.config.num_attention_heads
        kv_heads = self.config.num_key_value_heads
        dim_per_head = hidden_size // num_heads
        
        # Query, Key, Value projections
        # Query: (batch_size * seq_length, hidden_size) x (hidden_size, hidden_size)
        query_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, hidden_size)
        
        # Key and Value: Use num_key_value_heads for GQA
        key_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, 
                                                   dim_per_head * kv_heads)
        value_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, 
                                                     dim_per_head * kv_heads)
        
        # Reshape and transpose operations are just memory operations, 0 FLOPS
        
        # For each attention head
        per_head_flops = 0
        
        # QK^T operation: (batch_size * num_heads, seq_length, dim_per_head) x 
        # (batch_size * num_heads, dim_per_head, seq_length)
        qkt_flops = batch_size * num_heads * self.matrix_multiply_flops(seq_length, dim_per_head, seq_length)
        
        # Scale QK^T: seq_length * seq_length elements, 1 multiplication each
        scale_flops = batch_size * num_heads * seq_length * seq_length
        
        # Softmax:
        # exp for each element: 10 FLOPS * batch_size * num_heads * seq_length * seq_length
        # sum: batch_size * num_heads * seq_length * (seq_length - 1)
        # division: batch_size * num_heads * seq_length * seq_length
        softmax_flops = (batch_size * num_heads * seq_length * seq_length * 10) + \
                        (batch_size * num_heads * seq_length * (seq_length - 1)) + \
                        (batch_size * num_heads * seq_length * seq_length)
        
        # Attention x Value: (batch_size * num_heads, seq_length, seq_length) x 
        # (batch_size * num_heads, seq_length, dim_per_head)
        attn_value_flops = batch_size * num_heads * self.matrix_multiply_flops(seq_length, seq_length, dim_per_head)
        
        # Combine heads: (batch_size * seq_length, num_heads * dim_per_head) x (num_heads * dim_per_head, hidden_size)
        combine_heads_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, hidden_size)
        
        per_head_flops = qkt_flops + scale_flops + softmax_flops + attn_value_flops
        
        total_flops = query_proj_flops + key_proj_flops + value_proj_flops + \
                      per_head_flops + combine_heads_flops
                      
        return total_flops
    
    def mlp_flops(self):
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        seq_length = self.config.seq_length
        batch_size = self.config.batch_size
        
        # First linear layer split into two for SwiGLU
        # (batch_size * seq_length, hidden_size) x (hidden_size, intermediate_size)
        gate_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, intermediate_size)
        # (batch_size * seq_length, hidden_size) x (hidden_size, intermediate_size)
        up_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, hidden_size, intermediate_size)
        
        # SwiGLU activation
        swiglu_flops = self.swiglu_flops(batch_size * seq_length * intermediate_size)
        
        # Down projection
        # (batch_size * seq_length, intermediate_size) x (intermediate_size, hidden_size)
        down_proj_flops = self.matrix_multiply_flops(batch_size * seq_length, intermediate_size, hidden_size)
        
        return gate_proj_flops + up_proj_flops + swiglu_flops + down_proj_flops
    
    def embedding_flops(self):
        # Adding positional embeddings: batch_size * seq_length * hidden_size additions
        pos_embedding_flops = self.config.batch_size * self.config.seq_length * self.config.hidden_size
        
        return pos_embedding_flops
    
    def transformer_layer_flops(self):
        hidden_size = self.config.hidden_size
        seq_length = self.config.seq_length
        batch_size = self.config.batch_size
        
        # Input normalization (RMSNorm)
        input_norm_flops = self.layer_norm_flops(batch_size * seq_length * hidden_size)
        
        # Self-attention block
        attention_flops = self.self_attention_flops()
        
        # Residual connection: 1 addition per element
        residual1_flops = batch_size * seq_length * hidden_size
        
        # Pre-MLP normalization (RMSNorm)
        pre_mlp_norm_flops = self.layer_norm_flops(batch_size * seq_length * hidden_size)
        
        # MLP block
        mlp_flops = self.mlp_flops()
        
        # Residual connection: 1 addition per element
        residual2_flops = batch_size * seq_length * hidden_size
        
        return input_norm_flops + attention_flops + residual1_flops + \
               pre_mlp_norm_flops + mlp_flops + residual2_flops
    
    def final_layer_flops(self):
        hidden_size = self.config.hidden_size
        seq_length = self.config.seq_length
        batch_size = self.config.batch_size
        vocab_size = self.config.vocab_size
        
        # Final normalization
        final_norm_flops = self.layer_norm_flops(batch_size * seq_length * hidden_size)
        
        # Unembedding / Language modeling head
        # Only calculate for the last token position as we only care about next token prediction
        lm_head_flops = self.matrix_multiply_flops(batch_size * 1, hidden_size, vocab_size)
        
        return final_norm_flops + lm_head_flops
    
    def forward_pass_flops(self):
        # Embedding
        embedding_flops = self.embedding_flops()
        
        # Transformer layers
        layers_flops = self.config.num_hidden_layers * self.transformer_layer_flops()
        
        # Final layer
        final_flops = self.final_layer_flops()
        
        return embedding_flops + layers_flops + final_flops
    
    def training_step_flops(self):
        forward_flops = self.forward_pass_flops()
        backward_flops = 2 * forward_flops
        
        return forward_flops + backward_flops
    
    def inference_flops(self):
        return self.forward_pass_flops()
    
    def training_epoch_flops(self, num_samples: int):
        return self.training_step_flops() * (num_samples // self.config.batch_size)
    
    def lora_training_step_flops(self, lora_rank: int, num_lora_modules: int):
        
        # Forward pass is the same as regular forward pass
        forward_flops = self.forward_pass_flops()
        
        # For each LoRA module, we have two small matrices: A (d_in × r) and B (r × d_out)
        hidden_size = self.config.hidden_size
        
        # - A matrix: hidden_size × lora_rank parameters
        # - B matrix: lora_rank × hidden_size parameters
        lora_params_per_module = 2 * hidden_size * lora_rank
        
        seq_length = self.config.seq_length
        batch_size = self.config.batch_size
        layers = self.config.num_hidden_layers
        
        qv_params = 2 * layers * hidden_size * hidden_size  # Q and V matrices
        
        # Scale the backward FLOPS by the ratio of parameters
        lora_params = num_lora_modules * lora_params_per_module
        backward_scale = lora_params / qv_params
        
        backward_flops = int(2 * forward_flops * backward_scale)
        
        return forward_flops + backward_flops
    
    def summary(self, training: bool = True, lora: bool = False, lora_rank: int = 4, 
                num_lora_modules: int = None):
        
        if num_lora_modules is None:
            # By default, assume LoRA is applied to Q and V in each layer
            num_lora_modules = 2 * self.config.num_hidden_layers
            
        result = {
            "embedding_flops": self.embedding_flops(),
            "attention_flops_per_layer": self.self_attention_flops(),
            "mlp_flops_per_layer": self.mlp_flops(),
            "layer_norm_flops": self.layer_norm_flops(self.config.batch_size * self.config.seq_length * self.config.hidden_size),
            "transformer_layer_flops": self.transformer_layer_flops(),
            "final_layer_flops": self.final_layer_flops(),
            "forward_pass_flops": self.forward_pass_flops(),
        }
        
        if training:
            if lora:
                result["training_step_flops"] = self.lora_training_step_flops(lora_rank, num_lora_modules)
                result["lora_parameters"] = num_lora_modules * 2 * self.config.hidden_size * lora_rank
            else:
                result["training_step_flops"] = self.training_step_flops()
        else:
            result["inference_flops"] = self.inference_flops()
            
        return result


## Qwen2.5-0.5B-Instruct config parameters
def get_qwen_0_5b_config(manual_config):
    
    model, _ = load_qwen()
    config = model.config

    config.max_position_embeddings = manual_config['seq_length']
    config.num_hidden_layers = manual_config['hidden_layers']
     
    return ModelConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        group_query_attention=True,
        num_key_value_heads=config.num_key_value_heads,  # 16 / 8
        seq_length=config.max_position_embeddings,
        batch_size=manual_config['batch_size']
    )


def calculate_total_flops_experiment(manual_config, lora=False, is_training=True):
    
    config = get_qwen_0_5b_config(manual_config)
    
    counter = FLOPSCounter(config)
    
    if is_training:
        if lora:
            # 2 modules per layer: query and value projections
            num_lora_modules = 2 * config.num_hidden_layers
            step_flops = counter.lora_training_step_flops(manual_config['lora_rank'], num_lora_modules)
        else:
            step_flops = counter.training_step_flops()
        
        total_flops = step_flops * manual_config['training_steps']
    else:
        # For inference, calculate per token generation
        # Typically we would run a forward pass with the context, then generate each token
        context_flops = counter.forward_pass_flops()
        
        # For each additional token, we do a forward pass with a sequence length of 1
        # But we need to keep the context in attention
        token_gen_config = ModelConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            group_query_attention=config.group_query_attention,
            num_key_value_heads=config.num_key_value_heads,
            seq_length=1,  # Only generating one token at a time
            batch_size=manual_config['batch_size']
        )
        token_counter = FLOPSCounter(token_gen_config)
        token_flops = token_counter.forward_pass_flops()
        
        # Total FLOPS for context + generated tokens
        tokens_to_generate = manual_config['training_steps']  # Repurpose training_steps as tokens to generate
        total_flops = context_flops + (token_flops * tokens_to_generate)
    
    return total_flops


def print_experiment_flops_table(experiments):
    
    for exp in experiments:
        config, lora, training = exp  # Unpacking tuple

        total_flops = calculate_total_flops_experiment(
            manual_config=config,
            lora=lora,
            is_training=training
        )

        print(f"Model-Size: {config['model_size']:<12} "
              f"Seq_len: {config['seq_length']:<14} "
              f"Batch_size: {config['batch_size']:<12} "
              f"Lora_rank: {config['lora_rank']:<12} "
              f"Training_Steps: {config['training_steps']:<14} "
              f"Total_Flops: {total_flops:,}")  # Adds comma formatting for large FLOPS values


if __name__ == "__main__":
    # Example usage
    
    manual_config = {
        'seq_length' : 512,
        'batch_size' : 4,
        'lora_rank' : 4,
        'hidden_layers' : 24,
        'model_size' : '0.5b',
        'training_steps' : 5000
    }

    print('Maual-Config-Selected: ')
    print()
    pprint(manual_config)
    print()

    # Get config for 0.5B model with sequence length 512
    config = get_qwen_0_5b_config(manual_config)
    
    # Create FLOPS counter
    counter = FLOPSCounter(config)
    
    # Get FLOPS summary for regular training
    regular_summary = counter.summary(training=True, lora=False)
    
    # Get FLOPS summary for LoRA training with rank 4
    lora_summary = counter.summary(training=True, lora=True, lora_rank=manual_config["lora_rank"])
    
    # Print summaries
    print("\nRegular Training FLOPS Summary:")
    for key, value in regular_summary.items():
        print(f"{key}: {value:,}")
    
    print()

    print("\nLoRA Training FLOPS Summary:")
    for key, value in lora_summary.items():
        print(f"{key}: {value:,}")
    
    # Example experiments for FLOPS table
    ### Per train-step (foward + Backward) = 1.3 x 10^12 FLOPS (at max_tokens = 512, lora_rank = 4)
    experiments = [
    [{"model_size": "0.5b", "seq_length": 128, "batch_size": 4, "lora_rank": 4, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 128, "batch_size": 4, "lora_rank": 8, "training_steps": 500, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 8, "training_steps": 500, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 768, "batch_size": 4, "lora_rank": 8, "training_steps": 500, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 768, "batch_size": 4, "lora_rank": 8, "training_steps": 10000, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 4, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 4, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, False, True],
    [{"model_size": "0.5b", "seq_length": 768, "batch_size": 4, "lora_rank": 4, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True], 
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 2, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 4, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True],
    [{"model_size": "0.5b", "seq_length": 512, "batch_size": 4, "lora_rank": 8, "training_steps": 5000, 'hidden_layers' : manual_config['hidden_layers']}, True, True]
    ]

    print("\nFLOPS for Different Experiments:")
    print_experiment_flops_table(experiments)