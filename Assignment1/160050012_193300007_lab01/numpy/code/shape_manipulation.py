import numpy as np
import sys

file = sys.argv[1]

#### Assuming the numpy array is in csv format
arr = np.genfromtxt(file,delimiter=',',dtype=float)
print("M = ",end=" ")
m = int(input())
print("N = ",end=" ")
n = int(input())

answer = []

for i in range(arr.shape[0]):
    row = []
    for j in range(arr.shape[1]):
        value = arr[i][j]
        matrix = np.full((m,n),value)
        row.append(matrix)
    answer.append(row)
        
answer = np.asarray(answer)
print(answer)