import cv2
import numpy as np 
import matplotlib.pyplot as plt


##### Image path 
image = "../data/obelisk.png"

##### Reading Image
img = cv2.imread(image)
width , height ,channels = img.shape

##### Matplotlib to take coordinates of 4 corners
# rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(rgb)
# plt.show()

upperLeft = [220,241] 
lowerLeft = [704,1028]
lowerRight = [992,826]
upperRight = [481,186]
middle = [453,625]
originalC = np.array([upperLeft,lowerLeft,lowerRight,upperRight],dtype='float32')
newUpperLeft = [0,384]
newUpperRight = [0,0]
newLowerLeft =  [511,384]
newLowerRight = [511,0]
newMiddle = [256,384]
newC = np.array([newUpperLeft,newLowerLeft,newLowerRight,newUpperRight],dtype='float32')
updateArray = (newC-originalC)/8
count = 8
oldC = originalC
cv2.imshow("Progression",img)
while count > 0:
    newIntermediateC = oldC + updateArray
    matrix = cv2.getPerspectiveTransform(oldC,newIntermediateC)
    maxWidth = np.max(newIntermediateC,axis=0)[0]
    maxHeight = np.max(newIntermediateC,axis=0)[1]
    result = cv2.warpPerspective(img,matrix,(maxWidth,maxHeight))
    cv2.imshow("Progression",result)
    img = result    
    count = count - 1
    oldC = newIntermediateC
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
# #### Transformation Matrix for shear transformation
# M = np.array([[1,-0.1,0],[-0.1,1,0]],dtype='float32')
# #### New Image after affine transformation
# newImage = cv2.warpAffine(img,M,(600,600))
# ##### Displaying image
# cv2.imshow("Original Image - Manual",newImage)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
