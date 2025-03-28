import numpy as np
import matplotlib.pyplot as plt
import transformers
import os
import random
import h5py
import pandas as pd
import time
import importlib
from pprint import pprint
import torch 
from tqdm import tqdm
import re
import gc
from torch.utils.data import TensorDataset, DataLoader
import yaml

from forecast import *
from lora import LoRALinear
from preprocess import *
from data_create import *
from data_prepare import *
from qwen import *

# Import Accelerator
from accelerate import Accelerator

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print('Device-activated: ', device)

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "../data/lotka_volterra_data.h5")
file_path = os.path.abspath(file_path)  # Ensure it's absolute  

config_path = os.path.join(script_dir, "config.yaml")

save_dir = os.path.join(script_dir, "../saves")
save_dir = os.path.abspath(save_dir)

checkpoint_dir = os.path.join(save_dir, 'checkpoint')
plot_dir = os.path.join(save_dir, 'images') 

# Define LoRATrainer class
class LoRATrainer():
    def __init__(self, config_path=config_path):
        with open(config_path) as file:
            manual_config = yaml.safe_load(file)
        
        self.train_split = manual_config['train_split']
        self.time_step_split = manual_config['time_step_split']
        self.batch_size = manual_config['batch_size']
        self.learning_rate = manual_config['learning_rate']  # 1e-5
        self.lora_rank = manual_config['lora_rank']
        self.max_ctx_length = manual_config['seq_length']
        self.forecast_length = manual_config['forecast_length']
        self.max_tokens = manual_config['max_tokens']
        self.seq_length = manual_config['seq_length']
        self.hidden_layers = manual_config['hidden_layers']
        self.target_steps = manual_config['training_steps']
        self.val_limit = manual_config['val_limit']

        _, self.tokenizer = load_qwen()
        
    def get_model(self):
        ft = time.time()
        model_lora, _ = load_qwen()
        lt = time.time()

        print('Time taken to load model: ', (lt - ft) / 60, 'mins') 

        model_lora.config.max_position_embeddings = self.seq_length
        model_lora.config.num_hidden_layers = self.hidden_layers

        for layer in model_lora.model.layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=self.lora_rank) 
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=self.lora_rank)

        def get_model_params(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params

        total, trainable = get_model_params(model_lora)
        print(f"Total Parameters: {total:,}")
        print(f"Trainable Parameters: {trainable:,}")  # Should be significantly less

        pprint(model_lora.config)

        return model_lora

    def data_load_prepare(self):
        ### No test-train split because of chunking later
        data_prey, data_prey_true, data_pred, data_pred_true = load_data(file_path, self.time_step_split, is_plot=False)
        print('LOG: (data-load-shape)')
        print(data_prey.shape, data_prey_true.shape, data_pred.shape, data_pred_true.shape)

        train_input_ids, val_input_ids, self.prey_os, self.pred_os = prepare_data(data_prey, data_pred, self.tokenizer, self.max_ctx_length, self.train_split)
        print('LOG: (prepare-load-shape)')
        print(train_input_ids.shape,  val_input_ids.shape)

        train_dataset = TensorDataset(train_input_ids)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(val_input_ids)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader 

    def train(self):
        # Initialize Accelerator for device management
        accelerator = Accelerator()

        train_loader, val_loader = self.data_load_prepare()
        model_lora = self.get_model()

        optimizer = torch.optim.Adam((p for p in model_lora.parameters() if p.requires_grad), lr=self.learning_rate)

        model_lora.train()

        print('Target-Train-Steps:', self.target_steps)

        train_steps = 0
        progress_bar = tqdm(range(self.target_steps), desc="Training Steps")

        train_curve, val_curve = [], []

        # Prepare the model and optimizer for Accelerator
        model_lora, optimizer, train_loader, val_loader = accelerator.prepare(model_lora, optimizer, train_loader, val_loader)

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        # For best model based on validation loss
        best_val_loss = float('inf')
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

        if self.target_steps > 20:
            check_freq = self.target_steps // 2 ## Reducing save-checkpoints
        else:
            check_freq = 5

        if self.target_steps > 20:
            plot_freq = self.target_steps // 20 ## Reducing save-checkpoints
        else:
            plot_freq = 5

        ft = time.time()

        while train_steps < (self.target_steps - 1):
            
            for (batch_input_ids, ) in train_loader:
                
                optimizer.zero_grad()
                outputs = model_lora(batch_input_ids, labels=batch_input_ids)  # Use target_ids
                loss = outputs.loss  # Loss function is a model attribute
                loss.backward()
                optimizer.step()

                train_curve.append(loss.detach().cpu().item())  # Store loss for monitoring

                train_steps += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                '''
                Checkpointing - Memory bottleneck (max capped at 2)
                '''
                # Save checkpoint based on frequency (every 'check_freq' steps in this case)
                # if train_steps % check_freq == 0:
                #     checkpoint_path = os.path.join(checkpoint_dir, f"lora_step_{train_steps}.pt")
                    
                #     # Save LoRA adapter weights
                #     model_lora.save_pretrained(checkpoint_path)
                    
                #     # Save optimizer state
                #     opt_path = os.path.join(checkpoint_dir, f"optimizer_step_{train_steps}.pt")
                #     try:
                #         torch.save(optimizer.state_dict(), opt_path)
                #         print(f"Checkpoint saved at step {train_steps}")

                #     except (OSError, RuntimeError, IOError) as e:
                #         print(f"Warning: Could not save file {opt_path}. Continuing execution...")
                #         print(f"Error: {e}")

                if train_steps % plot_freq == 0:

                    fig = plt.figure(figsize=(7, 7))

                    ### loss progession tracking
                    plt.plot(range(len(train_curve)), train_curve, color='red', marker='.', label='Train')
                    plt.plot(range(len(val_curve)), val_curve, color='blue', marker='.', label='Validation')

                    plt.ylabel('Loss')
                    plt.xlabel('#Optimization Steps')

                    plt.title('Loss-Curve')

                    plt.legend()
                    plt.grid()

                    fn_name = f'inter_loss_curve_t_{train_steps}.png' 
                    plt.savefig(os.path.join(plot_dir, fn_name))
                    
                    plt.show()

                ### Validation Loop
                model_lora.eval()
                val_losses = []

                with torch.no_grad(): ### maybe - only validate on last batch (to save compute)
                    for b_n, (batch_input_ids, ) in enumerate(val_loader): ### LIMIT: val-limit - val-batches
                        if b_n >= self.val_limit:
                            break
                        val_op = model_lora(batch_input_ids, labels=batch_input_ids)
                        val_losses.append(val_op.loss.cpu().item())
                    
                    # Calculate average validation loss
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_curve.append(avg_val_loss)

                    # # Save best model based on validation loss
                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     model_lora.save_pretrained(best_model_path)
                    #     print(f"New best model saved with validation loss: {best_val_loss:.4f}")

                if train_steps >= self.target_steps:  # Stop training at the required steps
                    break

                model_lora.train()  # Resume training mode

        # Save final model after training
        final_model_path = os.path.join(save_dir, "final_model.pt")
        model_lora.save_pretrained(final_model_path)

        lt = time.time()
        print('Time taken:', (lt - ft) / 60, 'mins')

        return model_lora.eval(), train_curve, val_curve, self.target_steps


if __name__ == '__main__':
    print('Running __lora_train.py__ ....')

    # Initialize LoRATrainer and start training
    lora = LoRATrainer()
    _, train_curve, val_curve, target_steps = lora.train()

    # Plotting the loss curves
    plt.plot(range(len(train_curve)), train_curve, color='red', marker='.', label='Train')
    plt.plot(range(len(val_curve)), val_curve, color='blue', marker='.', label='Validation')

    plt.ylabel('Loss')
    plt.xlabel('#Optimization Steps')

    plt.title('Loss-Curve')

    plt.legend()
    plt.grid()

    # Save plot
    fn_name = f'final_loss_curve_s_{target_steps}.png'
    plt.savefig(os.path.join(save_dir, fn_name))

    plt.show()
