# Generative Adversarial Attacks on Image Classification



## How to Run
1. Set up a venv environment: `python3.11 -m venv tapod_env`
2. Install the requirements: `pip install -r requirements.txt`
3. Skip to the desired section (training or inference)

### Training
1. Run the desired script, corresponding to the desired level of complexity: 
    * Main: `python3 main/full.py`

### Inference: 
1. Run the desired script, corresponding to the desired level of complexity. 
    * Baseline: `python3 baseline/cls_full.py`
    * Main: `python3 main/eval.py`
        * Note: make sure the last two lines in `main/full.py` are commented first. 