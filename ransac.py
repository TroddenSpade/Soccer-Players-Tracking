import numpy as np
import cv2
import glob

sift = cv2.xfeatures2d.SIFT_create() # opencv 3
# use "sift = cv2.SIFT()" if the above fails

I2 = cv2.imread('field.jpg')
G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
G2 = cv2.GaussianBlur(G2,(5,5),0)
keypoints2, desc2 = sift.detectAndCompute(G2, None); # opencv 3

I1 = cv2.imread('bg.png')
G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
G1 = cv2.GaussianBlur(G1,(7,7),0)
keypoints1, desc1 = sift.detectAndCompute(G1, None); # opencv 3

# brute-force matching
bf = cv2.BFMatcher()

# for each descriptor in desc1 find its
# two nearest neighbours in desc2
matches = bf.knnMatch(desc1,desc2, k=2)

# distance ratio test
alpha = 0.8
good_matches = [m1 for m1,m2 in matches if m1.distance < alpha *m2.distance]

points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
points1 = np.array(points1,dtype=np.float32)

points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
points2 = np.array(points2,dtype=np.float32)

H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,2.0)

# H = np.eye(3,dtype=np.float32) # this needs to be changed
(m, n) = G1.shape
pts = np.float32([ [0,0],
                    [0,m],
                    [n,m],
                    [n,0] ]).reshape(-1,1,2) # this needs to be changed

dst = cv2.perspectiveTransform(pts,H).reshape(4,2)

J = I2.copy()
# cv2.line(J, (dst[0,0], dst[0,1]), (dst[1,0], dst[1,1]), (255,0,0),3)
# cv2.line(J, (dst[1,0], dst[1,1]), (dst[2,0], dst[2,1]), (255,0,0),3)
# cv2.line(J, (dst[2,0], dst[2,1]), (dst[3,0], dst[3,1]), (255,0,0),3)
# cv2.line(J, (dst[3,0], dst[3,1]), (dst[0,0], dst[0,1]), (255,0,0),3)


I = cv2.drawMatches(I1,keypoints1,J,keypoints2,good_matches, None)

cv2.imshow('keypoints',I)
cv2.waitKey()    
