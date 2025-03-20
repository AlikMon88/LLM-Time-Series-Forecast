import numpy as np
import re
import random

def create_forecast_prompt_sep(encoded_series, forecast_length=10, name='prey'):
    
    prompt = f"""<|im_start|>user
    I have a time series representing {name} populations. Each timestep is separated by commas.

    Time series data: {encoded_series}

    Predict exactly {forecast_length} timesteps, following the same pattern and formatting.
    Ensure the output consists of exactly {forecast_length} values, comma-separated, without any additional text or explanation.
    Stop generating after {forecast_length} steps.

    Output format example:
    [value1, value2, ..., value{forecast_length}]

    Strictly follow this format and do not generate more or fewer than {forecast_length} values.
    <|im_end|>
    <|im_start|>assistant
    """
    
    return prompt

def create_forecast_prompt_joint(encoded_series_prey, encoded_series_predator, forecast_length=10, prey_name='prey', predator_name='predator'):
    
    series_length = len(encoded_series_prey.split(', '))
    
    prompt = f"""<|im_start|>user
    I have time series for {prey_name} and {predator_name} populations.

    The data is formatted as: { '; '.join([f'{prey}, {pred}' for prey, pred in zip(encoded_series_prey.split(', '), encoded_series_predator.split(', '))]) };

    Forecast the next {forecast_length} points in the same format:
    {'; '.join([f'{{prey_{series_length + i}}}, {{pred_{series_length + i}}}' for i in range(1, forecast_length + 1)])}

    Generate exactly {forecast_length} pairs IN THE GIVEN FORMAT.
    <|im_end|>
    <|im_start|>assistant
    """

    # prompt = f"""
    # The data is formatted as: { '; '.join([f'{prey}, {pred}' for prey, pred in zip(encoded_series_prey.split(', '), encoded_series_predator.split(', '))]) };
    # Predict next {forecast_length} pairs in same format - prey_1, predator_1; prey_2, predator_2; prey_3, predator_3 ...
    # """

    return prompt

def extract_forecasts(forecast_output, model_type='llama'):
    """
    Given forecast_output in the format:
    0.477, 0.524;\n    0.32, 0.425;\n    0.259, 0.347; ...
    This function extracts and returns two lists: one for prey forecasts and one for predator forecasts.
    """

    # Remove any explicit '\n' escape sequences and extra spaces
    cleaned_output = forecast_output.replace('\\n', '').strip()

    # Regular expression pattern to match pairs of numbers (handles floats and integers)
    pattern = r'([-+]?\d*\.\d+|\d+)\s*,\s*([-+]?\d*\.\d+|\d+)(?=\s*;|$)'
    matches = re.findall(pattern, cleaned_output)

    prey_forecasts = [match[0] for match in matches]
    predator_forecasts = [match[1] for match in matches]

    if model_type == 'llama':
        prey_forecasts = ', '.join(prey_forecasts)
        predator_forecasts = ', '.join(predator_forecasts)

    elif model_type == 'gpt':
        prey_forecasts = ", ".join(" ".join(prey_forecasts))
        predator_forecasts = ", ".join(" ".join(predator_forecasts))

    return prey_forecasts, predator_forecasts


# Generate forecasts
def generate_forecast(model, encoded_series, tokenizer, forecast_length=10, max_new_tokens=100):

    prompt = create_forecast_prompt_sep(encoded_series, forecast_length)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set parameters for more deterministic generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,  # Low temperature for more deterministic output
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the newly generated tokens
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

if __name__ == '__main__':
    print('... __forecast.py__ ...')
