import os
import json
import yaml
import torch
import matplotlib.pyplot as plt

# Import the LoRATrainer from your existing script
from __lora_train__ import LoRATrainer

script_dir = os.path.dirname(os.path.abspath(__file__))

result_path = '../saves/hyper/hyperparameter_results.json'
result_path = os.path.join(script_dir, result_path)
result_path = os.path.abspath(result_path)

final_config_path = os.path.join(script_dir, '../saves')
final_config_path = os.path.abspath(final_config_path)    

def train_final_model():
    # Load best hyperparameters from previous search
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get best configuration
    best_config = results[0]
    best_config_path = best_config['config_path']
    
    # Load the best configuration
    with open(best_config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update training steps for final training
    config['training_steps'] = 30000
    
    # Save updated config
    with open(final_config_path, 'w') as file:
        yaml.dump(config, file)
    
    print("\n--- Final Model Training ---")
    print(f"Best Configuration:")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Context Length: {best_config['context_length']}")
    
    # Initialize trainer with best configuration
    lora_trainer = LoRATrainer(config_path=final_config_path)
    
    # Train final model
    final_model, train_curve, val_curve, target_steps = lora_trainer.train()
    
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
    plt.savefig(f'../saves/final_model_loss_curve_{target_steps}.png')
    plt.close()
    
    print("\nFinal model training completed. Model and loss curve saved.")

if __name__ == '__main__':
    train_final_model()