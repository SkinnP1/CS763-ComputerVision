import numpy as np
import cv2

def click_event1(event, x, y, flags, param):
    if event==cv2.EVENT_FLAG_LBUTTON:
        points1.append([x,y])
        radius = 8
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(img1, (x,y), radius, color, thickness)
        cv2.imshow("Image 1",img1)


points1 = []
cv2.namedWindow("Image 1",cv2.WINDOW_NORMALWINDOW_NORMAL)
img1 = cv2.imread("../data/calib_images/image3.jpg")
cv2.imshow("Image 1",img1)
cv2.setMouseCallback('Image 1', click_event1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("../data/calib_images/true.jpg",img1)
print(points1)