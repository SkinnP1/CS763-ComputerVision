import argparse
import cv2
import numpy as np 



def detectPoints(vertices):
	distance = np.sum(np.square(vertices), axis=2)
	minDistanceIndex = np.argmin(distance, axis=0)
	order = []
	count = 0
	while count != 4 :
		order.append(vertices[minDistanceIndex][:][:])
		minDistanceIndex = (minDistanceIndex+1)%4
		count = count + 1
	reOrder = np.array(order,dtype='float32')
	reOrder =  reOrder.reshape((4,1,2))
	return reOrder



parser = argparse.ArgumentParser()
parser.add_argument("-i", action="store", dest="image")
args = parser.parse_args()

image = args.image

#### Show image 
img = cv2.imread(image)
newimg = cv2.imread(image)
newimg1 = cv2.imread(image)
cv2.imshow("Original Image",img)
print("Press any key to continue")
cv2.waitKey(0)
cv2.destroyAllWindows()


# Image smoothing # Detecting edges using canny 
blur = cv2.GaussianBlur(img,(5,5), cv2.BORDER_DEFAULT)
canny = cv2.Canny(blur,100,150)
edge = cv2.GaussianBlur(canny,(3,3), cv2.BORDER_DEFAULT)

##### Drawing largest contour
contours , _ = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
areas = [cv2.contourArea(i) for i in contours]
c = contours[np.argmax(areas)]
cv2.drawContours(img,[c],-1,(255,0,0),5)
cv2.imshow("Contour with largest area",img)
print("Press any key to continue")
cv2.waitKey(0)
cv2.destroyAllWindows()


##### Detecting end vertices of polygon
vertices = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
### Drawing Convex Hull. We need convex quadrilateral
convex = cv2.convexHull(vertices)
##### Checking if the quadrilateral detected is convex or not
vertices = cv2.approxPolyDP(convex, 0.01 * cv2.arcLength(c, True), True)
if len(vertices) != 4 :
    print("No quadrilateral detected")
    exit(0)

###### Drawing bounding box around the detected contour
cv2.drawContours(newimg,[vertices],-1,(255,0,0),5)
cv2.imshow("Forming Convex Hull across largest area",newimg)
print("Press any key to continue")
cv2.waitKey(0)
cv2.destroyAllWindows()

###### Detect upper left, upper right, lower left, lower righ from vertices. Draw Orthographic Projection
vertices = vertices.astype('float32')
v = detectPoints(vertices)
final = np.array([[0,0],[599,0],[599,749],[0,749]],dtype='float32')
matrix = cv2.getPerspectiveTransform(v,final)
result = cv2.warpPerspective(newimg1,matrix,(600,750))
cv2.imshow("Final Orthographic View", result)
print("Press any key to continue")
cv2.waitKey(0)
cv2.destroyAllWindows()


####### Saving Image in results directory 
imageName = image.split('/')[-1]
onlyImageName = imageName.split(".")[0]
cv2.imwrite("../results/"+onlyImageName+"-output.jpg",result)


