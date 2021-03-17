import numpy as np 
import cv2
import argparse



im1 = cv2.imread("../data/calib_images/image1.jpg")
im2 = cv2.imread("../data/calib_images/image2.jpg")
true = cv2.imread("../data/true.jpg")

imagePoints1 = np.array([[1513, 730], [1381, 753], [1245, 779], [1543, 851], [1410, 876], [1274, 906]],dtype=np.float32)
imagePoints2 = np.array([[1686, 913], [1552, 882], [1420, 847], [1712, 789], [1577, 754], [1447, 723]],dtype=np.float32)
ref = np.array([[0,0,0],[17,0,0],[34,0,0],[0,17,0],[17,17,0],[34,17,0]],dtype=np.float32) #in mm


##### Configuring Object points in sequence.
objectPoints = np.array([ref,ref],dtype=np.float32)

imagePoints = np.array([imagePoints1,imagePoints2],dtype=np.float32)

print("Object Points:") 
print(objectPoints)
print()
#### Printing points
print("Points for image 1:")
print(imagePoints1)
print()
print("Points for image 2:")
print(imagePoints2)
print()
#### Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints,imagePoints,(4000,3000), None, None,flags=(cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6+cv2.CALIB_ZERO_TANGENT_DIST))


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



###### Draw x on reprojected points of first image
points, _ = cv2.projectPoints(objectPoints[0], rvecs[0], tvecs[0], mtx, dist)
points = np.array(points,dtype=np.float32)
points = points.reshape(6, 2)

print("Original points :")
print(imagePoints[0])
print()
print("Re projected points :")
print(points)
print()

mean_error = 0
for i in range(0,1):
    imgpoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2 = np.array(imgpoints2,dtype=np.float32)
    k = imgpoints2.reshape(6, 2)
    im1 = np.array(imagePoints[i],dtype=np.float32)
    error = np.sqrt(np.sum(np.square(im1 - k)))/6
    mean_error += error

print ("Error: ", mean_error)
print()


for i in points:
    cv2.drawMarker(true, (i[0],i[1]),(255,0,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2, line_type=cv2.LINE_AA)
cv2.namedWindow("Projected Points",cv2.WINDOW_NORMAL)
cv2.imshow("Projected Points",true)
print("Press any key to exit and store image")
cv2.waitKey(0)
cv2.imwrite("../data/projected.jpg",true)