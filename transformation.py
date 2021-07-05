import cv2
import numpy as np


def get_perspective_transform(type):
    global points1, points2
    if type == 1:
        points1 = np.array([(1223, 744.7),
                            (1225.44, 87.5),
                            (25, 148.2),
                            (224.35, 830)]).astype(np.float32)

        points2 = np.array([(367.28, 371),
                            (592.5, 42.55),
                            (41.55, 42.55),
                            (263.9, 371)]).astype(np.float32)

    if type == 0:
        points1 = np.array([(52, 309),
                            (588.66, 664.79),
                            (1236, 314),
                            (818, 159.5)]).astype(np.float32)

        points2 = np.array([(40.5, 371.5),
                            (211, 371.5),
                            (256, 206),
                            (40.5, 42.5)]).astype(np.float32)

    if type == 2:
        points1 = np.array([(402, 901),
                            (1257, 272),
                            (519, 177),
                            (26, 423)]).astype(np.float32)

        points2 = np.array([(317, 371),
                            (594.5, 371),
                            (594.5, 40.5),
                            (317, 236)]).astype(np.float32)

    for i in range(4):
        cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)
        cv2.circle(I2, (int(points2[i, 0]), int(points2[i, 1])), 3, [0, 0, 255], 2)

    # compute homography from point correspondences
    return cv2.getPerspectiveTransform(points1, points2)


I1 = cv2.imread('bg_1.png')
I2 = cv2.imread('field.jpg')

output_size = (I2.shape[1], I2.shape[0])
H = get_perspective_transform(1)
J = cv2.warpPerspective(I1, H, output_size)

cv2.imshow('I1', I1)
cv2.waitKey(0)

cv2.imshow('I2', I2)
cv2.waitKey(0)

cv2.imshow('J', J)
# cv2.imwrite('trans_1.png', J)
cv2.waitKey(0)
