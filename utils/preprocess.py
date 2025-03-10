import numpy as np


def ts_encoding(series, model_type="gpt", precision=3, alpha=0.99, beta=0.3):
    
    """
    Preprocess time series data for LLMs (GPT-style or LLaMA-style tokenization).
    
    Parameters:
    - series (list or np.array): Raw time series data.
    - model_type (str): "gpt" for GPT-style tokenization, "llama" for LLaMA-style.
    - precision (int): Number of decimal places to retain.
    - alpha (float): Percentile scaling for normalization.
    - beta (float): Offset for normalization.
    
    Returns:
    - str: Tokenized time series string.
    """

    # Scale the series
    series = np.array(series)
    min_val, max_val = np.min(series), np.max(series) ## Min-Max Scaling
    offset = min_val - beta * (max_val - min_val)
    scale_factor = np.percentile(series - offset, alpha * 100)

    print('Offset: ', offset, 'Scale-Factor: ', scale_factor)

    # Prevent division by zero
    if scale_factor == 0:
        scale_factor = 1

    # Normalize the series
    normalized_series = (series - offset) / scale_factor

    # Convert to fixed-precision strings
    formatted_values = [f"{x:.{precision}f}" for x in normalized_series]

    if model_type == "gpt":
        # GPT-style tokenization: Add spaces between digits and use commas between values
        tokenized_series = ", ".join(" ".join(str(x)) for x in formatted_values)

    elif model_type == "llama":
        # LLaMA-style tokenization: No spaces between digits, just comma-separated numbers
        tokenized_series = ", ".join(formatted_values)

    else:
        raise ValueError("model_type must be 'gpt' or 'llama'")

    return tokenized_series, offset, scale_factor


def ts_decoding(tokenized_series, model_type="gpt", precision=3, offset=0, scale_factor=1):
    """
    Convert tokenized LLM output back into numerical time series.

    Parameters:
    - tokenized_series (str): The string output from the LLM.
    - model_type (str): "gpt" for GPT-style, "llama" for LLaMA-style.
    - precision (int): Number of decimal places to round.
    - offset (float): Offset used during normalization (default=0, should match encoding step).
    - scale_factor (float): Scale factor used during normalization (default=1, should match encoding step).

    Returns:
    - np.array: Reconstructed time series values.
    """

    if model_type == "gpt":
        # GPT-style decoding: Remove spaces and split by commas
        values = tokenized_series.split(", ")
        values = ["".join(x.split()) for x in values]  # Remove spaces between digits

    elif model_type == "llama":
        # LLaMA-style decoding: Just split by commas (no extra spaces to remove)
        values = tokenized_series.split(", ")

    else:
        raise ValueError("model_type must be 'gpt' or 'llama'")

    # Convert to float
    decoded_series = np.array([float(x) for x in values])

    # Reverse normalization (scale and shift back)
    original_series = decoded_series * scale_factor + offset

    return np.round(original_series, precision)  # Ensure correct precision


if __name__ == '__main__':

    print('running ... __preprocess.py__ ... now')
        
    # Example Usage
    time_series_data = [0.123, 1.23, 12.3, 123.0]

    print("GPT-style Tokenization:")
    print(ts_encoding(time_series_data, model_type="gpt"))

    print("\nLLaMA-style Tokenization:")
    print(ts_encoding(time_series_data, model_type="llama"))