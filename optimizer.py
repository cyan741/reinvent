import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from data_structs import MolData
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from help.chem import *
from scoring_functions import tanimoto, logP
import json
from utils import weighted_geometric_mean
class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.query_structure = args.query_structure
            self.max_oracle_calls = args.max_oracle_calls
            self.seed =  args.seed
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)


    def log_intermediate(self, mols=None, scores=None, finish=False, seed = None):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
            print(f"length of molbuffter when ended:{len(self.mol_buffer)}")
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls

            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # try:
        print({
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
        })
        if finish:
            data = {
                "progress": f'{n_calls}/{self.max_oracle_calls}',
                "metrics": {
                    "avg_top1": avg_top1,
                    "avg_top10": avg_top10,
                    "avg_top100": avg_top100,
                    "avg_sa": avg_sa,
                    "diversity": diversity_top100
                },
                "auc": {
                    "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
                    "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
                    "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls)
                }
            }
            output_file = os.path.join(self.args.output_dir, 'results_' +  self.query_structure + "_" + str(seed)+".json")
            with open(output_file, 'w') as outfile:
                json.dump(data, outfile, indent=4)

    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            
            else:

                alpha = 10.0 
                score_qed = float(self.evaluator(smi))
                t = tanimoto(self.query_structure)
                score_tanimoto = np.array(t.__call__([smi]))[0]
                beta = 10.0
                gamma = 1.0
                s= logP()
                score_logp = np.array(s.__call__([smi]))[0]
                values = [score_logp, score_tanimoto, score_qed]
                weights = [alpha, beta, gamma]
                score = weighted_geometric_mean(values, weights)


                self.mol_buffer[smi] = [score, len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        self.smi_file = args.smi_file
        self.max_oracle_calls = args.max_oracle_calls
        self.oracle = Oracle(args=self.args)
        self.query_structure = args.query_structure
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False, seed = None):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish, seed=seed)
    
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        records = []
        for compound, (score, index) in self.mol_buffer.items():
            records.append({compound: [float(score), index]})
        
        with open(output_file_path, 'w') as f:
            yaml.dump(records, f, sort_keys=False)
    
    def _analyze_results(self, seed):
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

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
        results = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        sim_pass = 0
        logP_pass = 0
        qed_pass = 0
        sim_min = 1
        sim_max = 0
        qed_min = 1
        qed_max = 0
        for smi in smis:
            m = Chem.MolFromSmiles(smi)
            q = structures.get(self.query_structure)
            mq = Chem.MolFromSmiles(q)
            t = tanimoto(self.query_structure)
            score_tanimoto = np.array(t.__call__([smi]))
            if sim_min > score_tanimoto:
                sim_min = score_tanimoto

            if sim_max < score_tanimoto:
                sim_max = score_tanimoto
            s= logP()
            score_logp = np.array(s.__call__([smi]))
            qed_mq = Descriptors.qed(mq)
            qed_m = Descriptors.qed(m)
            if qed_min > qed_m:
                qed_min = qed_m
            if qed_max < qed_m:
                qed_max = qed_m
            if score_tanimoto > 0.6:
                sim_pass += 1
            if score_logp == 1.0:
                logP_pass += 1
            if qed_m > qed_mq:
                qed_pass += 1

        
        print(f"seed:{seed}")
        output_file = os.path.join(self.args.output_dir, 'results_' +  self.query_structure + "_" + str(seed)+".json")

        with open (output_file, "r") as infile:
            existing_data = json.load(infile)
        existing_data ["top100_pass_rates"]= {
            "sim_pass_0.6": float(sim_pass / 100),
            "sim_low_bound": float(sim_min),
            "sim_max_bound": float(sim_max),
            "logP_pass": float(logP_pass / 100),
            "qed_pass": float(qed_pass / 100), 
            "qed_low_bound": float(qed_min),
            "qed_max_bound": float(qed_max)
        }
        with open(output_file, 'w') as outfile:
            json.dump(existing_data, outfile, indent=4)
    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
    
    def optimize(self, oracle, config, query_structure, seed, project="test"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.oracle.task_label = self.model_name + "_" + query_structure + "_" + str(seed)
        self._optimize(oracle, config, query_structure, seed)
        self._analyze_results(seed)

        self.save_result(self.model_name + "_" + query_structure + "_" + str(seed))

        self.reset()

        
            
    def hparam_tune(self, oracles, hparam_space, hparam_default, count=5, num_runs=3, project="tune"):
        #seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19,
        seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        seeds = seeds[:num_runs]
        hparam_space["name"] = hparam_space["name"]
        
        def _func():
            avg_auc = 0
            for oracle in oracles:
                auc_top10s = []
                for seed in seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    self._optimize(oracle, self.config, self.query_structure, seed)
                    auc_top10s.append(top_auc(self.oracle.mol_buffer, 10, True, self.oracle.freq_log, self.oracle.max_oracle_calls))
                    self.reset()
                avg_auc += np.mean(auc_top10s)
            print({"avg_auc": avg_auc})
            


    def production(self, oracle, config, num_runs=5, project="production"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if num_runs > len(seeds):
            raise ValueError(f"Current implementation only allows at most {len(seeds)} runs.")
        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, self.query_structure, seed, project)
            self.reset()
