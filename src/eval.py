import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

from forecast import *
from preprocess import *

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
        
        print(predicted.shape, true.shape)
        
        metrics[m]['MAE'] = mean_absolute_error(true, predicted)
        metrics[m]['MSE'] = mean_squared_error(true, predicted)
        metrics[m]['RMSE'] = np.sqrt(metrics[m]['MSE'])
        metrics[m]['MAPE'] = np.mean(np.abs((true - predicted) / true)) * 100
        
        # Normalized RMSE
        metrics[m]['NRMSE'] = metrics[m]['RMSE'] / (np.max(true) - np.min(true))
        
        # R² Score
        metrics[m]['R2'] = r2_score(true, predicted)
    
    return metrics

def visualize_forecast_comparison(true_data, predicted_data, time_data, rn):
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

    plt.xlabel('time')
    plt.title('Prey-Predator-Population (Forecast) (Pred-Cut)')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_lora_modelevaluate_lora_model(model_lora, data_test, data_true, time_data, offset_scale, tokenizer, inf_max_token=21, is_instruction=False):
    
    pred_batch = []
    
    prey_os, pred_os = offset_scale
    prey_os, pred_os = prey_os[-len(data_test):], pred_os[-len(data_test):]
    
    for i, test_prompt in enumerate(data_test):
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
    visualize_forecast_comparison([data_true[0][rn], data_true[-1][rn]], [pred_batch[0][rn], pred_batch[-1][rn]], time_data, rn)
    
    return metrics


def main():
    
    # Load best model and configuration
    lora_trainer = LoRATrainer()
    model, _, _, _ = lora_trainer.train()
    
    # Load test data (you'll need to implement this based on your data preparation)
    test_data_prey, test_data_prey_true, test_data_pred, test_data_pred_true = load_data(file_path, time_step_split, is_plot=False)
    
    # Evaluate model
    metrics = evaluate_lora_model(
        model, 
        test_data_prey, 
        lora_trainer.tokenizer, 
        lora_trainer.max_ctx_length
    )
    
    # Print and save metrics
    import json
    print("Model Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    
    with open('../saves/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()