from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors

import time
import pickle
import re
import threading
import pexpect
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a single SMILES of
   argument and returns a float. A multiprocessing class will then spawn workers and divide the
   list of SMILES given between them.
"""


class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    
    #query_structure = "C1(S(N)(=O)=O)=CC=C(N2C(C3=CC=C(C)C=C3)=CC(C(F)(F)F)=N2)C=C1"#塞来昔布
    
    def __init__(self,query_structure):

        structures = {
            "Celebrex" : "C1(S(N)(=O)=O)=CC=C(N2C(C3=CC=C(C)C=C3)=CC(C(F)(F)F)=N2)C=C1",
            "Osimertinib" : "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC",
            "Fexofenadine" : "CC(C)(C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O)C(=O)O",
            "Ranolazine" : "CC1=C(C(=CC=C1)C)NC(=O)CN2CCN(CC2)CC(COC3=CC=CC=C3OC)O",
            "Perindopril": "CCCC(C(=O)O)NC(C)C(=O)N1C2CCCCC2CC1C(=O)O",
            "Amlodipine": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
            "Sitagliptin": "C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N",
            "Zaleplon": "CCN(C1=CC=CC(=C1)C2=CC=NC3=C(C=NN23)C#N)C(=O)C"

            }
        self.query_structure = structures.get(query_structure)
        if self.query_structure:
            query_mol = Chem.MolFromSmiles(self.query_structure)
            self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)
        else:
            print("can't find this structure")
    def __call__(self, smiles):
        mol_list = [Chem.MolFromSmiles(smile) for smile in smiles]
        resluts = []
        for mol in mol_list:
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
                score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
                score = min(score, self.k) / self.k
                #score = 2 * score -1
                resluts.append(score)
                
            else:
                resluts.append(0.0)
        return resluts


class logP():
    """Scores structures based on the logP of compound, the oracle function of logP has negtive values.
       If the logP of a mol falls into [0,3], the score is 1.0, else the score is 0.0.
    """
    def __init__(self,args=None):
        pass
    def __call__(self, smiles):
        mol_list = [Chem.MolFromSmiles(smile) for smile in smiles]
        resluts = []
        for mol in mol_list:
            if mol:
                logp = Descriptors.MolLogP(mol)
                if logp >= 0.0 and logp <= 3.0:
                    score = 1.0
                else:
                    score = 0.0
                resluts.append(score)
                
            else:
                resluts.append(0.0)
        return resluts



