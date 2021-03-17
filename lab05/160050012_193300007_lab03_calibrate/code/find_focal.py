import numpy as np 
import cv2


imagePoints1 = np.array([[1513, 730], [1381, 753], [1245, 779], [1543, 851], [1410, 876], [1274, 906]],dtype=np.float32)
imagePoints2 = np.array([[1686, 913], [1552, 882], [1420, 847], [1712, 789], [1577, 754], [1447, 723]],dtype=np.float32)
imagePoints3 = np.array([[2112, 1408], [1974, 1410], [1834, 1411], [2109, 1266], [1972, 1268], [1832, 1269]],dtype=np.float32)
ref = np.array([[0,0,0],[17,0,0],[34,0,0],[0,17,0],[17,17,0],[34,17,0]],dtype=np.float32) #in mm


imagePoints = np.array([imagePoints1,imagePoints2,imagePoints3],dtype=np.float32)
objectPoints = np.array([ref,ref,ref],dtype=np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints,imagePoints,(4000,3000), None, None,flags=(cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6+cv2.CALIB_ZERO_TANGENT_DIST))

###### Printing Focal Length
print("Calibration Matrix :")
print(mtx)
print()
print("Focal Length :")
print("fx in mm: ", mtx[0][0]*0.0008)
print("fy in mm : ", mtx[1][1]*0.0008)
print("Focal length of camera in mm : ", ((mtx[0][0]+mtx[1][1])/2)*0.0008)
print()

mean_error = 0
for i in range(len(objectPoints)):
    imgpoints2, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpoints2 = np.array(imgpoints2,dtype=np.float32)
    k = imgpoints2.reshape(6, 2)
    im1 = np.array(imagePoints[i],dtype=np.float32)
    error = np.sqrt(np.sum(np.square(im1 - k)))/18
    mean_error += error


print ("Average error: ", mean_error)
print()
