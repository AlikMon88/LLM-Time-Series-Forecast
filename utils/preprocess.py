import numpy as np
import torch

def ts_encoding(series, model_type="gpt", precision=3, alpha=0.99, beta=0.3):
    
    """
    Preprocess a batch of time series data for LLMs (GPT-style or LLaMA-style tokenization).
    
    Parameters:
    - series (np.array): Raw time series data with shape (n_samples, seq_len).
    - model_type (str): "gpt" for GPT-style tokenization, "llama" for LLaMA-style.
    - precision (int): Number of decimal places to retain.
    - alpha (float): Percentile scaling for normalization.
    - beta (float): Offset for normalization.
    
    Returns:
    - list of str: Tokenized time series strings.
    - np.array: Offsets for each sample.
    - np.array: Scale factors for each sample.
    """

    series = np.array(series)
    n_samples, seq_len = series.shape
    
    min_vals = np.min(series, axis=1, keepdims=True)
    max_vals = np.max(series, axis=1, keepdims=True)
    offsets = min_vals - beta * (max_vals - min_vals)
    scale_factors = np.percentile(series - offsets, alpha * 100, axis=1, keepdims=True)
    
    scale_factors[scale_factors == 0] = 1  # Prevent division by zero
    
    normalized_series = (series - offsets) / scale_factors
    formatted_values = np.round(normalized_series, precision).astype(str)
    
    tokenized_series = []
    for i in range(n_samples):
        if model_type == "gpt":
            tokenized_series.append(", ".join(" ".join(str(x)) for x in formatted_values[i]))
        elif model_type == "llama":
            tokenized_series.append(", ".join(formatted_values[i]))
        else:
            raise ValueError("model_type must be 'gpt' or 'llama'")
    
    return tokenized_series, offsets.squeeze(), scale_factors.squeeze()

def ts_decoding(tokenized_series, model_type="gpt", precision=3, offsets=None, scale_factors=None):
    
    """
    Convert a batch of tokenized LLM output back into numerical time series.
    
    Parameters:
    - tokenized_series (list of str): The list of tokenized time series strings.
    - model_type (str): "gpt" for GPT-style, "llama" for LLaMA-style.
    - precision (int): Number of decimal places to round.
    - offsets (np.array): Offsets used during normalization (shape: n_samples,).
    - scale_factors (np.array): Scale factors used during normalization (shape: n_samples,).
    
    Returns:
    - np.array: Reconstructed time series values with shape (n_samples, seq_len).
    """
    
    decoded_series = []
    for ts in tokenized_series:
        if model_type == "gpt":
            values = ts.split(", ")
            values = ["".join(x.split()) for x in values]  # Remove spaces between digits
        elif model_type == "llama":
            values = ts.split(", ")
        else:
            raise ValueError("model_type must be 'gpt' or 'llama'")
        
        decoded_series.append([float(x) for x in values])
    
    decoded_series = np.array(decoded_series)
    original_series = decoded_series * scale_factors[:, None] + offsets[:, None]
    
    return np.round(original_series, precision)

# Modified tokenization with chunking
def process_sequences(texts, tokenizer, max_length=512, stride=256):

    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        seq_ids = text.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
            
    return torch.stack(all_input_ids)


if __name__ == '__main__':

    print('running ... __preprocess.py__ ... now')
        
    # Example Usage
    time_series_data = [0.123, 1.23, 12.3, 123.0]

    print("GPT-style Tokenization:")
    print(ts_encoding(time_series_data, model_type="gpt"))

    print("\nLLaMA-style Tokenization:")
    print(ts_encoding(time_series_data, model_type="llama"))