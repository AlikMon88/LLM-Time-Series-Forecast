import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_forecasting_metrics(true_data, predicted_data):
    """
    Compute comprehensive forecasting metrics
    
    Args:
    true_data (np.ndarray): Ground truth time series data
    predicted_data (np.ndarray): Model predicted time series data
    
    Returns:
    dict: Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic Regression Metrics
    metrics['MAE'] = mean_absolute_error(true_data, predicted_data)
    metrics['MSE'] = mean_squared_error(true_data, predicted_data)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAPE'] = np.mean(np.abs((true_data - predicted_data) / true_data)) * 100
    
    # Normalized RMSE
    metrics['NRMSE'] = metrics['RMSE'] / (np.max(true_data) - np.min(true_data))
    
    # R² Score
    metrics['R2'] = r2_score(true_data, predicted_data)
    
    # Correlation Metrics
    metrics['Pearson_Correlation'], metrics['Pearson_P_Value'] = stats.pearsonr(true_data, predicted_data)
    metrics['Spearman_Correlation'], metrics['Spearman_P_Value'] = stats.spearmanr(true_data, predicted_data)
    
    # Phase Synchronization
    # Compute phase difference between true and predicted signals
    true_phase = np.unwrap(np.angle(np.fft.fft(true_data)))
    pred_phase = np.unwrap(np.angle(np.fft.fft(predicted_data)))
    metrics['Phase_Shift_Error'] = np.mean(np.abs(true_phase - pred_phase))
    
    # Forecast Skill Score
    # Compare against a naive forecast (e.g., mean or last value)
    naive_forecast = np.full_like(true_data, np.mean(true_data))
    naive_mse = mean_squared_error(true_data, naive_forecast)
    metrics['Forecast_Skill_Score'] = 1 - (metrics['MSE'] / naive_mse)
    
    return metrics

def visualize_forecast_comparison(true_data, predicted_data, save_path='../saves/forecast_comparison.png'):
    """
    Create visualization comparing true and predicted trajectories
    
    Args:
    true_data (np.ndarray): Ground truth time series data
    predicted_data (np.ndarray): Model predicted time series data
    save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true_data, label='True Trajectory', color='blue')
    plt.plot(predicted_data, label='Predicted Trajectory', color='red', linestyle='--')
    plt.title('Trajectory Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_lora_model(model, test_data, tokenizer, max_ctx_length):
    """
    Comprehensive model evaluation pipeline
    
    Args:
    model (torch.nn.Module): Trained LoRA model
    test_data (np.ndarray): Test dataset
    tokenizer: Tokenizer used for model
    max_ctx_length (int): Maximum context length
    
    Returns:
    dict: Comprehensive evaluation metrics
    """
    # Tokenize and prepare test data
    test_input_ids = prepare_input_for_inference(test_data, tokenizer, max_ctx_length)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model.generate(
            test_input_ids, 
            max_length=test_input_ids.shape[1] + 50,  # Generate additional steps
            num_return_sequences=1
        )
    
    # Decode predictions
    decoded_predictions = tokenizer.decode(predictions[0])
    
    # Convert decoded predictions back to numerical format
    predicted_data = convert_predictions_to_numerical(decoded_predictions)
    
    # Compute metrics
    metrics = compute_forecasting_metrics(test_data, predicted_data)
    
    # Visualize comparison
    visualize_forecast_comparison(test_data, predicted_data)
    
    return metrics

def main():
    # Example usage (you'll need to adapt to your specific data loading)
    from __lora_train__ import LoRATrainer
    
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