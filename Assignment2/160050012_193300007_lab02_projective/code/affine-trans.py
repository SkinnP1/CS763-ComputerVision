import argparse
import cv2
import numpy as np 
import matplotlib.pyplot as plt


##### Image path 
image = "../data/distorted.jpg"

##### Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-mat", action="store", dest="type")
args = parser.parse_args()

if args.type == 'manual':
    img = cv2.imread(image)
    #### Transformation Matrix for shear transformation
    M = np.array([[1,-0.1,0],[-0.1,1,0]],dtype='float32')
    #### New Image after affine transformation
    newImage = cv2.warpAffine(img,M,(600,600))
    ##### Displaying image
    cv2.imshow("Original Image - Manual",newImage)
    print("Press any key to continue")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    ##### Saving Image
    cv2.imwrite("../convincingDirectory/original-manual.jpg",newImage)
elif args.type == 'api':
    img = cv2.imread(image)
    #### Defining src and dest points (Used 3 points)
    src = np.array([[59,600],[660,660],[601,61]],dtype='float32')
    dest = np.array([[0,599],[599,599],[599,0]],dtype='float32')
    #### Transformation Matrix for shear transformation
    M = cv2.getAffineTransform(src,dest)
    #### New Image after transformation
    newImage = cv2.warpAffine(img,M,(600,600))
    ##### Displaying image
    cv2.imshow("Original Image - API",newImage)
    print("Press any key to continue")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    cv2.imwrite("../convincingDirectory/original-api.jpg",newImage)

