import os
import sys
import cv2
import numpy as np 



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

# Initialize the ORB detector algorithm 
orb = cv2.ORB_create(nfeatures=1000)
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
matches = bf.knnMatch(descriptors1, descriptors2,k=2)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)


if len(good) > 10:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    # Establish a homography
    M, L = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    result = warpImages(img2, img1, M)
    cv2.imshow("Panaromic Image",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("../results/pano-auto-results/panaromaSample.jpg",result)

    