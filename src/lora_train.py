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
from qwen import load_qwen

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print('Device-activated: ', device)
file_path = "data/lotka_volterra_data.h5"  # Change this to the correct path

class LoRATrainer():
    def __init__(self, config_path = 'config.yaml'):

        with open(config_path, "r") as file:
            manual_config = yaml.safe_load(file)
        
        self.train_split = manual_config['train_split']
        self.time_step_split = manual_config['time_step_split']
        self.batch_size = manual_config['batch_size']
        self.learning_rate = manual_config['learning_rate'] # 1e-5
        self.lora_rank = manual_config['lora_rank']
        self.max_ctx_length = manual_config['seq_length']
        self.forecast_length = manual_config['forecast_length']
        self.max_tokens = manual_config['max_tokens']
        self.seq_length = manual_config['seq_lenght']
        self.hidden_layers = manual_config['hidden_layers']
        self.target_steps = manual_config['training_steps']
        
    def get_model(self):
        
        ft = time.time()
        model_lora, self.tokenizer = load_qwen()
        lt = time.time()

        print('time-taken: ', (lt - ft)/60, 'mins') 

        pprint(model_lora.config)

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
        print(f"Trainable Parameters: {trainable:,}") ## 100x lesser

        return model_lora

    def data_load_prepare(self):
        
        ### NO: test-train split because of chunking later
        data_prey, data_prey_true, data_pred, data_pred_true = load_data(file_path, self.time_step_split, is_plot = False)
        print('LOG: (data-load-shape)')
        print(data_prey.shape, data_prey_true.shape, data_pred.shape, data_pred_true.shape)

        train_input_ids, train_target_ids, val_input_ids, val_target_ids, self.prey_os, self.pred_os = prepare_data(data_prey, data_pred, self.tokenizer, self.max_ctx_length, self.train_split)
        print('LOG: (prepare-load-shape)')
        print(train_input_ids.shape, train_target_ids.shape, val_input_ids.shape, val_target_ids.shape)

        train_dataset = TensorDataset(train_input_ids, train_target_ids)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(val_input_ids, val_target_ids)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader 

    def train(self):

        # Prepare components with Accelerator
        # accelerator = Accelerator()
        # model_lora, optimizer, train_loader, val_loader = accelerator.prepare(model_lora, optimizer, train_loader, val_loader)

        train_loader, val_loader = self.data_load_prepare()
        model_lora = self.get_model()
        
        optimizer = torch.optim.Adam((p for p in model_lora.parameters() if p.requires_grad), lr=self.learning_rate)

        model_lora.train()

        print('Target-Train-Steps:', self.target_steps)

        train_steps = 0
        progress_bar = tqdm(range(self.target_steps), desc="Training Steps")

        train_curve, val_curve = [], []

        # best_val_loss = float('inf')
        # checkpoint_freq = 5  # Save model every 5 steps - adjust this as needed

        # # Create checkpoint directory if it doesn't exist
        # checkpoint_dir = "model_checkpoints"
        # os.makedirs(checkpoint_dir, exist_ok=True)

        ft = time.time()

        while train_steps < self.target_steps:
            for batch_input_ids, batch_target_ids in train_loader:
                optimizer.zero_grad()
                outputs = model_lora(batch_input_ids, labels=batch_target_ids)  # Use target_ids
                loss = outputs.loss  # Loss function is a model attribute
                loss.backward()
                optimizer.step()

                train_curve.append(loss.detach().cpu().item())  # Store loss for monitoring

                train_steps += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                
                # # Save checkpoint based on frequency
                # if train_steps % checkpoint_freq == 0:
                #     checkpoint_path = os.path.join(checkpoint_dir, f"lora_step_{train_steps}.pt")
                #     # Save LoRA adapter weights
                #     model_lora.save_pretrained(checkpoint_path)
                #     # Save optimizer state
                #     torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"optimizer_step_{train_steps}.pt"))
                #     print(f"Checkpoint saved at step {train_steps}")

            
                ### FOR EVERY TRAINING-STEP WE RUN V-B BATCH 0(T_B * V_B)
                # Validation Loop

                model_lora.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_input_ids, batch_target_ids in val_loader:
                        val_op = model_lora(batch_input_ids, labels=batch_target_ids)
                        val_losses.append(val_op.loss.cpu().item())
                    
                    # Calculate average validation loss
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_curve.append(avg_val_loss)
                    
                    # # Save best model based on validation loss
                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    #     model_lora.save_pretrained(best_model_path)
                    #     print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
                if train_steps >= self.target_steps:  # Stop training at the required steps
                    break
                
                model_lora.train()  # Resume training mode
            
        # # Save final model
        # final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        # model_lora.save_pretrained(final_model_path)

        lt = time.time()
        print('Time taken:', (lt - ft) / 60, 'mins')

        return model_lora.eval(), train_curve, val_curve     



if __name__ == '__main__':
    
    print('Running __lora_train.py__ ....')

    lora = LoRATrainer()
    lora.train()
    


