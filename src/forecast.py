import numpy as np
import re
import random
import torch

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

def create_forecast_prompt_joint(encoded_series, forecast_length=10, prey_name='prey', predator_name='predator', is_show = False):
    
    series_len = len(encoded_series.split(';')) // 2
    
    prompt = f"""
    <|im_start|>user
    I have time series for {prey_name} and {predator_name} populations.

    The data is formatted as: {encoded_series};

    Predict the next {forecast_length} points in the same format as below:
    {'; '.join([f'{{prey_{series_len + i}}}, {{pred_{series_len + i}}}' for i in range(1, forecast_length + 1)])}

    JUST PREDICT DON'T SAY ANYTHING
    
    <|im_end|>
    <|im_start|>assistant
    """
    
    if is_show:
        print('PROMPT-ED:')
        print(prompt)
    return prompt

def create_forecast_prompt_joint_lora(encoded_series_prey, encoded_series_predator):

    prompt = f"""{ '; '.join([f'{prey}, {pred}' for prey, pred in zip(encoded_series_prey.split(', '), encoded_series_predator.split(', '))])};"""

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
def generate_forecast(model, prompt, tokenizer, max_new_tokens=100, temperature=0.1, is_tokenized=False):

    if not is_tokenized:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    else:
        inputs = {'input_ids': prompt.to(model.device)}

    # Set parameters for more deterministic generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,  # Low temperature for more deterministic output
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract only the newly generated tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def generate_forecast_v2(model, history, tokenizer, inf_max_new_tokens = 100, temperature=0.1):

    instruction = "Predict the next prey and predator populations based on the historical data."
    history_text = "".join(history)
    
    prompt = f"<|im_start|>user\n{instruction}\nHistorical data: {history_text}<|im_end|>\n<|im_start|>assistant\n"
    
    print('PROMPT-ED:')
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=inf_max_new_tokens, 
        temperature=temperature, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):]

def generate_forecast_v3(model, chunks, tokenizer, max_new_tokens, temperature=0.1, is_tokenized=True):
    
    '''
    Recursive Forecasting based on process_sequences chunking 
    '''   
    
    predictions = []
    context = None  # To keep track of the previous context
    
    for chunk_idx, chunk in enumerate(chunks):
        # If tokenized input is passed, use it
        if is_tokenized:
            inputs = {'input_ids': torch.unsqueeze(chunk, axis=0)}  # Assume chunk is already tokenized
        else:
            inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(model.device)

        # If this is not the first chunk, append previous predictions to the current input
        if context is not None:
            inputs['input_ids'] = torch.cat([context, inputs['input_ids']], dim=-1)

        # Generate the next tokens (future timesteps)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the newly generated tokens (after the original context)
        generated_sequence = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.append(generated_sequence)

        # Update context to include the newly generated tokens for the next chunk
        # For overlapping chunks, we include the last portion of the previous output as context for the next one
        context = outputs[:, -inputs['input_ids'].shape[-1]:]  # Keep only the last part of the chunk
        

    return predictions

if __name__ == '__main__':
    print('... __forecast.py__ ...')
