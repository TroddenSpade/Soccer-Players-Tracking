import cv2
import numpy as np

I1 = cv2.imread('bg.png')
I2 = cv2.imread('field.jpg')

n = 626
m = 417
output_size = (n, m)

points1 = np.array([(873, 780),
                    (1136, 115),
                    (640, 110),
                    (141, 168)]).astype(np.float32)

points2 = np.array([(317, 370),
                    (500, 92),
                    (317, 10),
                    (134, 94)]).astype(np.float32)

for i in range(4):
    cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)
    cv2.circle(I2, (int(points2[i, 0]), int(points2[i, 1])), 3, [0, 0, 255], 2)

# compute homography from point correspondences
H = cv2.getPerspectiveTransform(points1, points2)

output_size = (I2.shape[1], I2.shape[0])
J = cv2.warpPerspective(I1, H, output_size)

cv2.imshow('I1', I1)
cv2.waitKey(0)

cv2.imshow('I2', I2)
cv2.waitKey(0)

cv2.imshow('J', J)
cv2.waitKey(0)