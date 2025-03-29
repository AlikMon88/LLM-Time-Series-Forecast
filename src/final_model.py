import os
import json
import yaml
import torch
import matplotlib.pyplot as plt

# Import the LoRATrainer from your existing script
from lora_train import LoRATrainer

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, '../saves')
save_dir = os.path.abspath(save_dir)

result_path = 'hyper/hyperparameter_results.json'
result_path = os.path.join(save_dir, result_path)

image_path = os.path.join(save_dir, 'hyper/images') 

def train_final_model():
    
    with open(result_path, 'r') as file:
        best_config = json.load(file)
    file.close()
    
    best_config = best_config[0]
    best_config_path = best_config['config_path']
     
    # Load the best configuration
    with open(best_config_path, 'r') as file:
        best_config = yaml.safe_load(file)
    
    # Update training steps for final training
    best_config['training_steps'] = 15000 ## change to 30k
    
    # Save updated config
    with open(result_path, 'w') as file:
        yaml.dump(best_config, file)
    
    print("\n--- Final Model Training ---")
    print(f"Best Configuration:")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Context Length: {best_config['max_tokens']}")
    
    os.makedirs(image_path, exist_ok=True)
    
    # Initialize trainer with best configuration
    lora_trainer = LoRATrainer(config_path=best_config_path)
    
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
    plt.savefig(os.path.join(image_path, f'final_model_loss_curve_{target_steps}.png'))
    plt.close()
    
    print("\nFinal model training completed. Model and loss curve saved.")
    
    return final_model

if __name__ == '__main__':
    _ = train_final_model()