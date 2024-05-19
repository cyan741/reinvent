#!/bin/bash

python run.py reinvent --seed 1 --query_structure Osimertinib &
python run.py reinvent --seed 2 --query_structure Fexofenadine &
python run.py reinvent --seed 3 --query_structure Ranolazine &
python run.py reinvent --seed 4 --query_structure Perindopril &
python run.py reinvent --seed 5 --query_structure Amlodipine &
python run.py reinvent --seed 6 --query_structure Sitagliptin &
python run.py reinvent --seed 7 --query_structure Zaleplon &
wait
echo "所有命令执行完毕"
