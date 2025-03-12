import numpy as np
import torch

def ts_encoding(series, model_type="llama", precision=3, alpha=0.99, beta=0.3):
    
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
    n_samples = len(series)
    
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

def ts_decoding(tokenized_series, model_type="llama", precision=3, offsets=None, scale_factors=None):
    """
    Convert tokenized LLM output back into numerical time series for either single samples or batches.
    
    Parameters:
    - tokenized_series (str or list of str): Single tokenized time series string or list of tokenized strings.
    - model_type (str): "gpt" for GPT-style, "llama" for LLaMA-style.
    - precision (int): Number of decimal places to round.
    - offsets (float or np.array): Offset(s) used during normalization.
    - scale_factors (float or np.array): Scale factor(s) used during normalization.
    
    Returns:
    - np.array: Reconstructed time series values.
    """
    
    # Handle single sample case
    single_sample = False
    if isinstance(tokenized_series, str):
        tokenized_series = [tokenized_series]
        single_sample = True
        if offsets is not None and not isinstance(offsets, (int, float)):
            offsets = np.array([offsets])
        if scale_factors is not None and not isinstance(scale_factors, (int, float)):
            scale_factors = np.array([scale_factors])
    
    # Convert scalar values to arrays if needed
    if offsets is not None and isinstance(offsets, (int, float)):
        offsets = np.full(len(tokenized_series), offsets)
    if scale_factors is not None and isinstance(scale_factors, (int, float)):
        scale_factors = np.full(len(tokenized_series), scale_factors)
    
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
    
    # Apply denormalization if offsets and scale_factors are provided
    if offsets is not None and scale_factors is not None:
        original_series = decoded_series * scale_factors[:, None] + offsets[:, None]
    else:
        original_series = decoded_series
    
    result = np.round(original_series, precision)
    
    # Return single sample without batch dimension if input was a single sample
    if single_sample:
        return result[0]
    else:
        return result

# Modified tokenization with chunking
def process_sequences(texts, tokenizer, max_length=256, stride=128): #stride(128)-token overlap between consecutive chunks, helping models retain context better.

    all_input_ids = []
    for text in texts:

        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

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