# Targeted Adversarial Perturbations for Object Detectors (TAPOD)



## How to Run
1. Set up a venv environment: `python3.11 -m venv tapod_env`
2. Install the requirements: `pip install -r requirements.txt`
3. Skip to the desired section (training or inference)

### Training
1. Run the desired script, corresponding to the desired level of complexity: 
    1. Baseline: `python3 baseline/full.py`
    2. Main: `python3 main/full.py`
    3. Stretch: `python3 stretch/full.py`

### Inference: 
1. Run the desired script, corresponding to the desired level of complexity: 
    1. Baseline: `python3 baseline/eval.py`
    2. Main: `python3 main/eval.py`
    3. Stretch: `python3 stretch/eval.py`
        * Note: make sure the command that launches training in `stretch/full.py` is commented first. 