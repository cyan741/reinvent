# Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning

## Requirements
To build the environment:
```
conda create -n molopt python=3.7
conda activate molopt
pip install scikit_learn
pip install tqdm
pip install pexpect
pip install torch 
pip install PyTDC 
pip install PyYAML
conda install -c rdkit rdkit 
```
## Usage

To train a Prior using:

`./train_prior.py` 

The checkpoint file will be included in ./data/Prior.ckpt

To optimize molecules, use the run.py script. For example:

```
# specify multiple random seeds 
python run.py reinvent --seed 0 1 2
# run 5 runs with different random seeds 
python run.py reinvent --task production --n_runs 5 
```





