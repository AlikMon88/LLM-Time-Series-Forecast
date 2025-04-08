import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from pprint import pprint
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from forecast import *
from preprocess import *
from qwen import load_qwen
from data_create import *
from data_prepare import *


script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, "../data/lotka_volterra_data.h5")
file_path = os.path.abspath(file_path)  # Ensure it's absolute  

eval_dir_path = os.path.join(script_dir, "../saves/eval_dir")
eval_dir_path = os.path.abspath(eval_dir_path)

lora_model_path = os.path.join(script_dir, "../saves/lora_model.pt")
lora_model_path = os.path.abspath(lora_model_path)

def compute_forecasting_metrics(true_data, predicted_data):
    
    true_prey, true_pred = true_data
    predicted_prey, predicted_pred = predicted_data[:, 0, :], predicted_data[:, 1, :]
    
    metrics = {'Prey' : {}, 'Predator' : {}}
    data_true = {'Prey': true_prey, 'Predator': true_pred}
    data_predicted = {'Prey': predicted_prey, 'Predator': predicted_pred}
    
    for m in metrics:
        
        # Basic Regression Metrics
        predicted = data_predicted[m]
        true = data_true[m][:, :predicted.shape[-1]]
        
        
        metrics[m]['MAE'] = mean_absolute_error(true, predicted)
        metrics[m]['MSE'] = mean_squared_error(true, predicted)
        metrics[m]['RMSE'] = np.sqrt(metrics[m]['MSE'])
        metrics[m]['MAPE'] = np.mean(np.abs((true - predicted) / true)) * 100
        
        # Normalized RMSE
        metrics[m]['NRMSE'] = metrics[m]['RMSE'] / (np.max(true) - np.min(true))
        
    return metrics

def visualize_forecast_comparison(true_data, predicted_data, time_data):
    """
    Create visualization comparing true and predicted trajectories
    
    Args:
    true_data (np.ndarray): Ground truth time series data
    predicted_data (np.ndarray): Model predicted time series data
    save_path (str): Path to save the visualization
    """
    time_data_true = time_data[-1]
    
    plt.figure(figsize=(12, 6))
    
    # plt.plot(time_data_past, true_data[check_rn].tolist(), label = 'Past Data')
    plt.plot(time_data_true[:len(predicted_data[0])], predicted_data[0], label = 'Prediction (Prey)', marker = 'x')
    plt.plot(time_data_true[:len(predicted_data[0])], true_data[0].tolist()[:len(predicted_data[0])], label = 'Truth (Prey)', marker = '.')

    # plt.plot(time_data_past, data_pred[check_rn].tolist(), label = 'Past Data')
    plt.plot(time_data_true[:len(predicted_data[-1])], predicted_data[-1], label = 'Prediction (Predator)', marker = 'x')
    plt.plot(time_data_true[:len(predicted_data[-1])], true_data[-1].tolist()[:len(predicted_data[-1])], label = 'Truth (Predator)', marker = '.')

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Prey-Predator-Population (Forecast) (Pred-Cut)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(eval_dir_path, 'sample-metric-forecast'))
    plt.show()

def evaluate_lora_model(model_lora, data_test, data_true, time_data, offset_scale, tokenizer, inf_max_token=128, is_instruction=False):
    
    pred_batch = []
    
    prey_os, pred_os = offset_scale
    prey_os['offset'], prey_os['scale'] = prey_os['offset'][-len(data_test):], prey_os['scale'][-len(data_test):]
    pred_os['offset'], pred_os['scale'] = pred_os['offset'][-len(data_test):], pred_os['scale'][-len(data_test):]
    
    for i, test_prompt in enumerate(data_test):
        
        print('Sample-ID: ', i)
        
        if is_instruction:
            test_prompt = create_forecast_prompt_joint(test_prompt, forecast_length=21, is_show = True)
        
        prey_pred_response = generate_forecast(model_lora, test_prompt, tokenizer, max_new_tokens=inf_max_token, temperature=0.1)
        prey_decoded_response, pred_decoded_response = extract_forecasts(prey_pred_response)
        prey_decoded_response = ts_decoding(prey_decoded_response, model_type="llama", precision=3, offsets=prey_os['offset'][i], scale_factors=prey_os['scale'][i])
        pred_decoded_response = ts_decoding(pred_decoded_response, model_type="llama", precision=3, offsets=pred_os['offset'][i], scale_factors=pred_os['scale'][i])
        
        min_len = min(len(prey_decoded_response), len(pred_decoded_response))
        ### This can be inhomogenous because the min_len can be different for different generations
        # pred_batch.append([prey_decoded_response[:min_len], pred_decoded_response[:min_len]])
        pred_batch.append([prey_decoded_response, pred_decoded_response])
    
    ## for homogeneity
    min_len = min([min(len(data[0]), len(data[-1])) for data in pred_batch])      
    pred_batch = [[data[0][:min_len], data[-1][:min_len]] for data in pred_batch]
    pred_batch = np.array(pred_batch)
    metrics = compute_forecasting_metrics(data_true, pred_batch)
    
    rn = random.randint(0, len(pred_batch[0]) - 1)
    visualize_forecast_comparison([data_true[0][rn], data_true[-1][rn]], [pred_batch[0][rn], pred_batch[-1][rn]], time_data)
    
    return metrics


def main(model, tokenizer, model_type = 'baseline', val_limit=5):
    
    data_prey, data_prey_true, data_pred, data_pred_true, time_data_past, time_data_true = load_data(file_path, time_step_split=0.8, is_plot = True)
    print(data_prey.shape, data_prey_true.shape, data_pred.shape, data_pred_true.shape, time_data_past.shape, time_data_true.shape)
    
    time_data = [time_data_past, time_data_true]
    
    _, _, prey_os, pred_os, data_test = prepare_data(data_prey, data_pred, tokenizer, max_ctx_length=512, train_split=0.8, forecast_length=128, is_forecast=True)
    
    data_prey_true, data_pred_true = data_prey_true[-len(data_test):], data_pred_true[-len(data_test):]
    print(data_prey_true.shape, data_pred_true.shape)
    
    data_test, data_prey_true, data_pred_true = data_test[:val_limit], data_prey_true[:val_limit], data_pred_true[:val_limit]    
    print(data_test.shape, data_prey_true.shape, data_pred_true.shape)
    
    data_true = [data_prey_true, data_pred_true]
    
    offset_scale = [prey_os, pred_os]
    eval_metrics = evaluate_lora_model(model, data_test, data_true, time_data, offset_scale, tokenizer=tokenizer, is_instruction=False)
            
    pprint(eval_metrics)
    
    with open(os.path.join(eval_dir_path, f'eval_metrics_{model_type}.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    
    
if __name__ == '__main__':
    
    import transformers
    from transformers import AutoModelForCausalLM

    script_dir = os.path.dirname(os.path.abspath(__file__))

    
    parser = argparse.ArgumentParser(description="Evaluate a model using baseline or LoRA.")
    parser.add_argument('--model_type', choices=['baseline', 'lora'], required=True,
                        help="Choose 'baseline' for the base model or 'lora' for the fine-tuned LoRA model.")
    parser.add_argument('--lora_model_path', type=str, default=None,
                        help="Path to the fine-tuned LoRA model (required if model_type='lora').")
    parser.add_argument('--val_limit', type=str, required=True, default=5,
                        help="Validate on how many sammples max=200")
    
    args = parser.parse_args()

    print("Running __eval.py__...") 
    
    if args.model_type == 'baseline':
        model, tokenizer = load_qwen()
        
    elif args.model_type == 'lora':
        if not args.lora_model_path:
            raise ValueError("Please provide --lora_model_path when using model_type='lora'")
        
        lora_model_path = args.lora_model_path
        
        print('LORA-path: ', lora_model_path)
        
        _, tokenizer = load_qwen()
        model = AutoModelForCausalLM.from_pretrained(lora_model_path, trust_remote_code=True)
    
    else:
        print('Wrong Model Passed!')
    
    main(model, tokenizer, model_type=args.model_type, val_limit=int(args.val_limit))