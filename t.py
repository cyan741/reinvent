import numpy as np
import os
import sys
from scoring_functions import tanimoto
#from optimizer import Oracle
from utils import Variable, seq_to_smiles, unique
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

o = Oracle()
a = o.assign_evaluator(evaluator=tdc.Oracle(name="QED"))
s = o.score_smi(smi)
print(s)