import argparse
import sys
import numpy as np

args = sys.argv[1:]
pValue = 2
v = []

for i in range(len(args)):
    if args[i]== '--p':
        pValue = float(args[i+1])
    else :
        v.append(float(args[i]))
        
vector = np.abs(np.array(v))
newVector = np.power(vector,pValue)
sumValue = np.sum(newVector)
answer = np.linalg.norm(vector,pValue)

print("Norm of",v,"is",np.around(answer,2))