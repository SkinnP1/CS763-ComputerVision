import numpy as np 
import cv2
import argparse



def process(x):
    a = x.split(" ")
    if a[0] == "":
        return([])
    else :
        return ([float(a[0]),float(a[1])])


def carttopolar(x,y,x0=0,y0=0):
    x1=x-x0
    y1=y-y0

    t = np.arctan2(y1,x1)*180/np.pi
    if y1<0:
        t=360+t
    return t

def rearrange(l):
    (x0,y0)=np.mean(l,axis=0).tolist()

    l = sorted(l, key=lambda coord: carttopolar(coord[0], coord[1],x0,y0))
    k = np.argmin([np.sqrt(x1**2+y1**2) for x1,y1 in l])
    anticlockwise = l[k:] + l[:k] 
    a= anticlockwise[:3][::-1]
    b= anticlockwise[3:]
    return (a+b)

##### Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", action="store", dest="path")
args = parser.parse_args()
txt = args.path



###### Open and read txt file. Saved in list.
f = open(txt, "r")
points = []
for x in f:
    p = process(x)
    if len(p)>0:
        points.append(p)

##### Points in clockwise direction. Starting from leftmost corner point
image1 = rearrange(points[:6])
image2 = rearrange(points[6:])

##### Configuring Object points in sequence.
objectPoints = np.array([[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],
                        [[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]]],dtype=np.float32)

imagePoints = np.array([image1,image2],dtype=np.float32)


#### Printing points
print("Rearranged points for image 1:")
print(image1)
print()
print("Rearranged points for image 2:")
print(image2)
print()
#### Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints,imagePoints,(1000,1000), None, None,flags=(cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6+cv2.CALIB_ZERO_TANGENT_DIST))


##### Printing camera matrices
print("Intrinsic Camera matrix is :")
print(mtx)
print()
print("Rotation vectors are :")
print(rvecs)
print()
print("Translation vectors are :")
print(tvecs)
print()

mean_error = 0
for i in range(len(objectPoints)):
    imgpoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2 = np.array(imgpoints2,dtype=np.float32)
    k = imgpoints2.reshape(6, 2)
    im1 = np.array(imagePoints[i],dtype=np.float32)
    error = np.sqrt(np.sum(np.square(im1 - k)))/12
    mean_error += error


print ("Average error: ", mean_error)



