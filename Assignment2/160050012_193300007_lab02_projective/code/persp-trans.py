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


cv2.imshow("Progression",img)
cv2.waitKey(1000)

upperLeft = [220,241] 
lowerLeft = [704,1028]
lowerRight = [992,826]
upperRight = [481,186]
originalC = np.array([upperLeft,lowerLeft,lowerRight,upperRight])
stencil = np.zeros(img.shape).astype(img.dtype)
color = [255, 255, 255]
cv2.fillPoly(stencil, [originalC], color)
result = cv2.bitwise_and(img, stencil)
cv2.imshow("Progression", result)
cv2.waitKey(1000)
img = result

originalC = originalC.astype('float32')
newUpperLeft = [0,384]
newUpperRight = [0,0]
newLowerLeft =  [511,384]
newLowerRight = [511,0]
newC = np.array([newUpperLeft,newLowerLeft,newLowerRight,newUpperRight],dtype='float32')
updateArray = (newC-originalC)/9
count = 9
oldC = originalC

print("Press Q key to exit the animation")
while count > 0:
    newIntermediateC = oldC + updateArray
    matrix = cv2.getPerspectiveTransform(oldC,newIntermediateC)
    maxWidth = np.max(newIntermediateC,axis=0)[0]
    maxHeight = np.max(newIntermediateC,axis=0)[1]
    if count == 1:
        result = cv2.warpPerspective(img,matrix,(512,385))
        cv2.imwrite("../convincingDirectory/obelisk-output.png",result)
    else:
        result = cv2.warpPerspective(img,matrix,(maxWidth,maxHeight))
    cv2.imshow("Progression",result)
    img = result   
    count = count - 1
    oldC = newIntermediateC
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

print("Animation Ended. Press any key to close the window. ")
cv2.waitKey(0)
cv2.destroyAllWindows()

