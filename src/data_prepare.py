import numpy as np
from forecast import *
from preprocess import *
from data_create import *
import random

def prepare_data(data_prey, data_pred, tokenizer, max_ctx_length, train_split, is_forecast = False):

    encoded_prey, offset_prey, scale_prey = ts_encoding(data_prey, model_type="llama", precision=3, alpha=0.99, beta=0.3)
    encoded_pred, offset_pred, scale_pred = ts_encoding(data_pred, model_type="llama", precision=3, alpha=0.99, beta=0.3)

    prey_os = {'offset' : offset_prey,
               'scale': scale_prey}
    
    pred_os = {'offset' : offset_pred,
               'scale': scale_pred}

    prey_pred_encoded = []
    for token_prey, token_pred in zip(encoded_prey, encoded_pred):
        prey_pred_encoded.append(create_forecast_prompt_joint_lora(token_prey, token_pred))

    prey_pred_encoded = np.array(prey_pred_encoded)
    
    data_train, data_test = prey_pred_encoded[:int(train_split * len(prey_pred_encoded))], prey_pred_encoded[int(train_split * len(prey_pred_encoded)):]  
    data_test = data_test[:int(0.5*len(data_train))]

    train_input_ids, train_target_ids = preprocess_sequences_v2(data_train, tokenizer, max_ctx_length) ## Its past-future chunking almost 80 times within 100 nested iteration
    rn_idx = np.random.randint(0, len(train_input_ids), size = len(data_train))
    train_input_ids, train_target_ids = train_input_ids[rn_idx], train_target_ids[rn_idx] ## we overfit on a 1% subset 

    val_input_ids, val_target_ids = preprocess_sequences_v2(data_test, tokenizer, max_ctx_length)
    rn_idx = np.random.randint(0, len(val_input_ids), size = len(data_test))
    val_input_ids, val_target_ids = val_input_ids[rn_idx], val_target_ids[rn_idx]

    if is_forecast:
        train_input_ids, train_target_ids, val_input_ids, val_target_ids, prey_os, pred_os, prey_pred_encoded[random.randint(0, len(prey_pred_encoded))]
    else:
        return train_input_ids, train_target_ids, val_input_ids, val_target_ids, prey_os, pred_os
    