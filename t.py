import numpy as np
import os
import sys
from scoring_functions import tanimoto
#from optimizer import Oracle
from utils import Variable, seq_to_smiles, unique, weighted_geometric_mean
import tdc
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Draw
from scoring_functions import logP
from rdkit.Chem import Descriptors
from optimizer import Oracle
smi = "c1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F"
smi4 = "Cc1cccc(S(=O)(=O)NC(=O)c2cc(C)ccc2C)c1"
m = Chem.MolFromSmiles(smi)

score = 0

alpha = 1.0  
score_qed = 1
score_tanimoto = 1
beta = 10.0
gamma = 1.0
score_logp = 1
values = [score_qed, score_tanimoto, score_logp]
weights = [alpha, beta, gamma]
scores = [value * weight for value, weight in zip(values, weights)]
for v in scores:
    score += v
score = score / sum(weights)
print(score)