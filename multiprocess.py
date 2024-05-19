
import importlib
import sys
'''
This code is a simple script that takes the name of a scoring function from the command line arguments, 
dynamically imports that scoring function from a specified module, and enters an infinite loop. 
In each iteration, it reads a SMILES string from standard input, passes it to the scoring function 
for evaluation, and receives a floating-point score representing the given SMILES. If an exception 
occurs during the execution of the scoring function, it sets the score to 0.0. Finally, it writes 
the SMILES string, the score, and a newline character to standard output, and flushes the output 
buffer for timely display.
'''
scoring_function = sys.argv[1]
func = getattr(importlib.import_module("scoring_functions"), scoring_function)()

while True:
    smile = sys.stdin.readline().rstrip()
    try:
        score = float(func(smile))
    except:
        score = 0.0
    sys.stdout.write(" ".join([smile, str(score), "\n"]))
    sys.stdout.flush()



