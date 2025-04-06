import os
import json
import yaml
import torch
import matplotlib.pyplot as plt
import time
from pprint import pprint

# Import the LoRATrainer from your existing script
from lora_train import LoRATrainer
from flops_counter import *

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, '../saves')
save_dir = os.path.abspath(save_dir)

best_config_json = 'hyper/hyperparameter_results.json'
best_config_json = os.path.join(save_dir, best_config_json)

image_path = os.path.join(save_dir, 'hyper/images') 

def best_model_flops_counter(r_config):
    ### 30k Steps ==> 3.9e15
    
    print('Config-Selected: ')
    print()
    pprint(r_config)
    print()

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

def train_final_model():
    
    with open(best_config_json, 'r') as file:
        best_config = json.load(file)
    file.close()
    
    best_config = best_config[0]
    best_config_yaml = best_config['config_path']
    
    print('Best-Config-Path (.yaml): ', best_config_yaml)
        
    # Load the best configuration
    with open(best_config_yaml, 'r') as file:
        best_config = yaml.safe_load(file)
    
    # Update training steps for final training
    best_config['training_steps'] = 10000 ## change to 30k
    
    # Save updated config
    with open(best_config_yaml, 'w') as file:
        yaml.dump(best_config, file)
    
    print("\n--- Final Model Training ---")
    print(f"Best Configuration:")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Context Length: {best_config['max_tokens']}")
    
    os.makedirs(image_path, exist_ok=True)
    
    # Initialize trainer with best configuration
    lora_trainer = LoRATrainer(config_path=best_config_yaml)
    
    ft = time.time()
    
    # Train final model
    final_model, train_curve, val_curve, target_steps = lora_trainer.train()
    
    lt = time.time()
    
    print('Time taken:', (lt - ft) / 60, 'mins')
    
    final_model_path = os.path.join(save_dir, "best_final_model.pt")
    final_model.save_pretrained(final_model_path)

    # Create comprehensive loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='Train Loss', color='red', alpha=0.7)
    plt.plot(val_curve, label='Validation Loss', color='blue', alpha=0.7)
    plt.title('Final Model: Training and Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(image_path, f'best_model_loss_curve_{target_steps}.png'))
    plt.close()
    
    print("\nFinal model training completed. Model and loss curve saved.")
    
    return final_model, best_config

if __name__ == '__main__':
    _, best_config = train_final_model()
    best_model_flops_counter(best_config)