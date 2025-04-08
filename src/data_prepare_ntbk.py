import numpy as np
import random

from .forecast import *
from .preprocess import *
from .data_create import *

def prepare_data(data_prey, data_pred, tokenizer, max_ctx_length, train_split, forecast_length = 5, prep_overlap = True, is_forecast = False):

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
    # prey_pred_encoded = prey_pred_encoded[np.random.permutation(len(prey_pred_encoded))]
    
    data_train, data_test = prey_pred_encoded[:int(train_split * len(prey_pred_encoded))], prey_pred_encoded[int(train_split * len(prey_pred_encoded)):]  
    
    print()
    print('sample-train-encoded:')
    print(data_train[0][:30])
    print()
    
    if len(data_train) < len(data_test):
        data_test = data_test[:int(0.5*len(data_train))]


    if not prep_overlap:
        ''' process_sequences_v2 - has a localized context-window of 5 time-steps + fixed instruction tokenization '''
        train_input_ids, train_target_ids = preprocess_sequences_v2(data_train, tokenizer, forecast_length, max_ctx_length) ## Its past-future chunking almost 80 times within 100 nested iteration
        val_input_ids, val_target_ids = preprocess_sequences_v2(data_test, tokenizer, forecast_length, max_ctx_length)
    
    else:
        ''' process_sequences_v1 - Global and larger context window - better for hyoerparameter optimization '''
        train_input_ids = process_sequences(data_train, tokenizer, max_length=max_ctx_length, stride=max_ctx_length // 2)
        val_input_ids = process_sequences(data_train, tokenizer, max_length=max_ctx_length, stride=max_ctx_length)
    
    if is_forecast:
        return train_input_ids, val_input_ids, prey_os, pred_os, data_test
    else:
        return train_input_ids, val_input_ids, prey_os, pred_os


if __name__ == '__main__':
    prepare_data()