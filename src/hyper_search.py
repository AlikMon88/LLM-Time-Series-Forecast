import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import json
from pprint import pprint

from lora_train import LoRATrainer
from flops_counter import *

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
        
# Create directory for hyperparameter results
hyper_save_dir = os.path.join(script_dir, '../saves/hyper/')
hyper_save_dir = os.path.abspath(hyper_save_dir)

images_dir = os.path.join(hyper_save_dir, 'images')

### NEED: Count Flops for all these Operations -- 
def hyper_flops_counter(r_config):
    ## Each train step -- 1.3e12 --> 10K step 1.3e15 --> hyper_search --? 35.1e15 (3.5e16)
    
    config = get_qwen_0_5b_config(r_config)
    
    counter = FLOPSCounter(config)
    
    regular_summary = counter.summary(training=True, lora=False)
    
    lora_summary = counter.summary(training=True, lora=True, lora_rank=r_config["lora_rank"])
    
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
    [{"model_size": "0.5b", "seq_length": r_config['max_tokens'], "batch_size": r_config['batch_size'], "lora_rank": r_config['lora_rank'], "training_steps": r_config['training_steps'], 'hidden_layers' : r_config['hidden_layers']}, True, True]
    ]

    print()
    print("\nFLOPS Counted:")
    print_experiment_flops_table(experiments) 

    
def run_hyperparameter_search():
    
    # Hyperparameter grid
    learning_rates = [1e-5, 5e-5, 4e-4]
    lora_ranks = [2, 4, 8]
    context_lengths = [128, 512, 768]
    
    # learning_rates = [1e-5]
    # lora_ranks = [2]
    # context_lengths = [128]
    
    # Track results
    results = []
    
    os.makedirs(hyper_save_dir, exist_ok=True)
    
    # Hyperparameter search
    for lr, rank, ctx_len in product(learning_rates, lora_ranks, context_lengths):
        print(f"\n--- Running Hyperparameter Combination ---")
        print(f"Learning Rate: {lr}, LoRA Rank: {rank}, Context Length: {ctx_len}")
        
        # Modify config for each run
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update config with current hyperparameters
        config['learning_rate'] = lr
        config['lora_rank'] = rank
        config['seq_length'] = ctx_len
        config['max_tokens'] = ctx_len ## Context-len needs to optimized for 2k steps
        config['training_steps'] = 10000  # Fixed training steps for comparison (NEED: 10k steps)
        
        # Save modified config
        run_config_path = os.path.join(hyper_save_dir, f'config_lr{lr}_rank{rank}_ctx{ctx_len}.yaml')
        with open(run_config_path, 'w') as file:
            yaml.dump(config, file)
        
        # flops counting
        print()
        hyper_flops_counter(config)
        print()
        
        # Initialize trainer with modified config
        lora_trainer = LoRATrainer(config_path=run_config_path)
        
        try:
            # Train model
            _, train_curve, val_curve, steps = lora_trainer.train()
            
            # Calculate final validation loss
            final_val_loss = val_curve[-1] if val_curve else float('inf')
            
            # Store results
            results.append({
                'learning_rate': lr,
                'lora_rank': rank,
                'context_length': ctx_len,
                'final_validation_loss': final_val_loss,
                'config_path': run_config_path
            })
            
            # Plot and save loss curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_curve, label='Train Loss', color='red')
            plt.plot(val_curve, label='Validation Loss', color='blue')
            plt.title(f'Loss Curves (LR:{lr}, Rank:{rank}, Ctx:{ctx_len})')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.grid()
            plt.savefig(os.path.join(images_dir, f'loss_lr{lr}_rank{rank}_ctx{ctx_len}_s{steps}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in hyperparameter run: {e}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['final_validation_loss'])
    
    # Save results
    with open(os.path.join(hyper_save_dir, 'hyperparameter_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print and return best configuration
    best_config = results[0]
    print("\n--- Best Hyperparameter Configuration ---")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Context Length: {best_config['context_length']}")
    print(f"Validation Loss: {best_config['final_validation_loss']}")
    
    return best_config

if __name__ == '__main__':
    run_hyperparameter_search()