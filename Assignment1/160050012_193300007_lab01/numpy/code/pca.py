import numpy as np
import sys
import matplotlib.pyplot as plt

#File name
file = sys.argv[1]

#### Assuming the numpy array is in txt file with format mentioned
arr = np.genfromtxt(file,delimiter=',',dtype=float)
arr = (arr - np.mean(arr,axis=0))/np.std(arr,axis=0,ddof=1)
covariance = np.cov(arr.T,ddof=0)
values , vectors = np.linalg.eig(covariance)

##### Sorting eigen values in descending order
index = values.argsort()[::-1]
values = values[index] 
vectors = vectors[:,index]

transformed = arr.dot(vectors[:,:2])

## 1st column = X ,  2nd column = Y
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.scatter(transformed[:,0],transformed[:,1])
plt.xlabel("First Column of Transformed Matrix")
plt.ylabel("Second Column of Transformed Matrix")
plt.title("Scatter Plot for Transformed Array in 2 dimensions")
plt.savefig('../results/out.png')