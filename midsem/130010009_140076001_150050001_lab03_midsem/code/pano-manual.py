import os
import sys
import cv2
import numpy as np 


def click_event1(event, x, y, flags, param):
    if event==cv2.EVENT_FLAG_LBUTTON:
        points1.append([x,y])
        img1[y-2:y+2,x-2:x+2]=[0, 0, 255]
        cv2.imshow("Image 1",img1)
    
def click_event2(event, x, y, flags, param):
    if event==cv2.EVENT_FLAG_LBUTTON:
        points2.append([x,y])
        img2[y-2:y+2,x-2:x+2]=[0, 0, 255]
        cv2.imshow("Image 2",img2)

# def i1_i2(points1,points2,img1,img2):
#     src = np.array(points1,dtype='float32')
#     dest = np.array(points2,dtype='float32')
#     M = cv2.getPerspectiveTransform(src,dest)
#     val = img1.shape[1] + img2.shape[1]
#     result_image = cv2.warpPerspective(img1, M , (val , img1.shape[0]))
#     result_image[0:img2.shape[0], 0:img2.shape[1]] = img2
#     cv2.imshow("Panaromic Image - Image 2 as reference ",result_image)
#     print("Press any key to exit")
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows()
#     cv2.imwrite("../results/pano-manual-results/panaromaSample-I1reference.jpg",result_image)

def warpImages(img1,img2,H):
  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]
  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)
  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  translation_dist = [-x_min,-y_min]
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
  return (output_img)


path = sys.argv[1]
images =[]
for file in os.listdir(path):
    images.append(path+file)

img1 = cv2.imread(images[0])
img2 = cv2.imread(images[1])


points1 = []
points2 = []


img1 = cv2.resize(img1,(700,700))
img2 = cv2.resize(img2,(700,700))
im1 = img1.copy()
im2 = img2.copy()

cv2.namedWindow("Image 1",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image 1",(700,700))
cv2.imshow("Image 1",img1)
cv2.setMouseCallback('Image 1', click_event1)

cv2.namedWindow("Image 2",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image 2",(700,700))
cv2.moveWindow('Image 2',900,135)
cv2.imshow("Image 2",img2)
cv2.setMouseCallback('Image 2', click_event2)
print("Select points on each image")
print("Press any key to continue")
cv2.waitKey(0)
cv2.destroyAllWindows()

src = np.array(points1,dtype='float32')
dest = np.array(points2,dtype='float32')
M, L = cv2.findHomography(src, dest, cv2.RANSAC,5.0)
result = warpImages(im2, im1, M)
cv2.imshow("Panaromic Image - Image 1 as reference ",result)
print("Press any key to exit")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("../results/pano-manual-results/panaromaSample-I1reference.jpg",result)
