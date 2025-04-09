# LLM for Time-Series Forecasting

This repository contains my implementation of M2 Courework.

---

## Repository Structure

```plaintext
.
├── data/                          # Contains the data for the problem
|
├── log/                           # Logged Results
|
├── run/                           # Contains the SLURM script for CSD3 execution
│   
├── report/                        # Contains the coursework report
│   └── main.pdf
│
├── src/                           # source file - Python utility scripts for execution
│   
├── main_lora.ipynb                # py notebook for LoRA Adaption
|
├── main_baseline.ipynb            # py notebook for Zero-Shot Forecasting for Qwen 2.5
│
├── .gitignore                     # Git ignore file
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

# Setup Instructions
Follow the steps below to set up and run the project on your local machine:

## 1. Clone the Repository
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/am3353.git
cd am3353
```

## 2. Set Up the Python Environment
Create and activate a venv
```bash 
# Create a virtual environment
python -m venv m2-env

# Activate the virtual environment
# On Windows:
m2-env\Scripts\activate
# On MacOS/Linux:
source m2-env/bin/activate
```

## 3. Install Dependencies
Use the `requirements.txt` to install all the necessary libraries
```bash
pip install -r requirements.txt
```

## 4. Run the Notebooks
```bash
## Install Jupyter
pip install notebook ipykernel

## Register venv as a jupyter kernel 
python -m ipykernel install --user --name=m2-env --display-name "Python (m2-env)"

## Launch Jupyter Notebook
jupyter notebook
```
In the browser, navigate to ```main_*.ipynb```  to run it.

To deactivate the python env just run ```deactivate``` in the terminal

## Report
The detailed findings, methodology, and results are documented in the coursework report:
`report/main.pdf`

## Declaration of Use of Autogeneration Tools

I acknowledge the use of the following generative AI tools in this coursework:

1. **ChatGPT** (https://chat.openai.com/)  
Used for code refactoring, generating code snippets and boilerplate templates, and accelerating the debugging process.

2. **Perplexity AI** (https://www.perplexity.ai/)  
Used for general research and fact-checking via the "deep research" feature.

A declaration regarding the use of generative AI tools in writing the report is provided within the report itself.


## License
This project is licensed under the MIT License.
