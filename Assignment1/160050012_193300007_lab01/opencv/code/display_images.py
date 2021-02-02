import numpy as np
import cv2
import sys
import os


directory = sys.argv[1]

### Saving all images in directory
images = []
imageNames = []
for file in os.listdir(directory):
    img = cv2.imread(os.path.join(directory,file),cv2.IMREAD_COLOR)
    if img is not None:
        images.append(img)
        imageNames.append(file)

start = 0
size = len(images)

cv2.namedWindow("Images",0)
while True:
    cv2.imshow("Images",images[start])
    key = cv2.waitKey(5) ##Wait for 50 secs before any key is pressed
    if key == ord('p'):
        start = (start-1)%size
    elif key == 27 : ##Escape Key
        break
    elif key == ord('n'):
        start = (start+1)%size
cv2.destroyAllWindows()





