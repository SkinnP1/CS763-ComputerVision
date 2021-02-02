import argparse
import sys
import numpy as np


# Computing Original Array 

def originalArray(N):
    arr = np.zeros((N,N),dtype=float)
    row = 0
    col = 0
    for i in range(0,N):
        if row >= N or col >= N :
            break
        arr[row][col] = 1
        row = row + 1
        col = col + 2

    col = 1
    for i in range(0,N):
        if row >= N or col >= N :
            break
        arr[row][col] = 1
        row = row + 1
        col = col + 2

    return arr
    

# Cropping Image
def crop(arr,offsetH,offsetW,targetH,targetW):
    croppedArray = arr[offsetH:offsetH+targetH,offsetW:offsetW+targetW]
    return croppedArray

# Padding an Array
def paddedArray(arr):
    a = np.pad(arr,(2,2),'constant',constant_values = 0.5)
    return a


args = sys.argv[1:]

N = 2
for i in range(len(args)):
    if args[i]== '--N':
        N = int(args[i+1])

if N == 2:
    print("Enter N ")
    exit()

###Computing 1st part
arr = originalArray(N)
arr = np.array(arr)
arr = arr.astype('float64')

print("Original array:")
print(arr)
print()

##### Assuming some values for crop function
offsetH = 1
offsetW = 1
targetH = 2
targetW = 2

croppedArr = crop(arr,offsetH, offsetW, targetH, targetW)
print("Cropped array:")
print(croppedArr)
print()

newArr = paddedArray(croppedArr)
print("Padded array:")
print(newArr)
print()

joinedArr = np.hstack((newArr,newArr))
print("Concatenated array: shape=("+str(joinedArr.shape[0])+", " + str(joinedArr.shape[1]) + ")")
print(joinedArr)






