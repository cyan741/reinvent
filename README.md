
# a RL way for molecule optimization
## Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning


## Requirements

This package requires:
* Python 3.6
* PyTorch 0.1.12 
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Scikit-Learn (for QSAR scoring function)
* tqdm (for training Prior)
* pexpect
* PyTDC 
* PyYAML

## Usage

To train a Prior using:

`./train_prior.py` 
The checkpoint file will be included in ./data/Prior.ckpt

To optimize molecules, use the main.py script. For example:

* `./main.py --scoring-function activity_model --num-steps 1000`

Training can be visualized using the Vizard bokeh app. The vizard_logger.py is used to log information (by default to data/logs) such as structures generated, average score, and network weights.




