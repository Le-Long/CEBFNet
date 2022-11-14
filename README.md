# CEBFNet
This is the code base for the undergrad thesis "OUT-OF-SCOPE INTENT DETECTION WITH META-LEARNING"
The slide for the thesis can be seen [here](https://docs.google.com/presentation/d/1f6VhiseQ8vxfYsEuUFcbjwxVgHKx_wA4PyUlNjKUByY/edit?usp=sharing)

## Usage
### Prepare your environment 

Download required packages
```
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r ./requirements.txt
```
### Baseline
To run vanilla RoBERTa and DNNC scaling experiments, refer to [the original DNNC project](https://github.com/salesforce/DNNC-few-shot-intent). The code here only contains minimal edits.

### CEBFNet
The code for CEBFNet is modified from [codebase of NBFNet](https://github.com/DeepGraphLearning/NBFNet).
Run the CEBFNet model on OOS CLINC Banking dataset
```
python run_nbfnet.py -c clinc_banking.yaml --gpus [0]
```
If you see the model stop at the first epoch, clear the cache
```
rm -r ~/.cache/torch_extensions/*
```
