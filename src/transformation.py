import cv2
import numpy as np


def get_perspective_transform(type):
    global points1, points2
    if type == 1:
        points1 = np.array([(1255, 744.7), #br
                            (1225.44, 87.5), #tr
                            (25, 148.2), # tl
                            (230.35, 830)]).astype(np.float32) # bl

        points2 = np.array([(350.28, 371), #br
                            (592.5, 42.55), #tr
                            (41.55, 42.55), #tl
                            (260.9, 371)]).astype(np.float32) # bl

    if type == 0:
        points1 = np.array([(52, 309),
                            (977.66, 925.64),
                            (1254.5, 207.52),
                            (818, 159.5)]).astype(np.float32)

        points2 = np.array([(40.5, 371.5),
                            (295.5, 371.5),
                            (246.75, 40.5),
                            (40.5, 40.5)]).astype(np.float32)

    if type == 2:
        points1 = np.array([(338.34, 949.59),
                            (1257, 272),
                            (519, 177),
                            (11, 252.7)]).astype(np.float32)

        points2 = np.array([(302.23, 371),
                            (594.5, 371),
                            (594.5, 35.5),
                            (344.52, 42.5)]).astype(np.float32)

    # for i in range(4):
    #     cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)
    #     cv2.circle(I2, (int(points2[i, 0]), int(points2[i, 1])), 3, [0, 0, 255], 2)

    # compute homography from point correspondences
    return cv2.getPerspectiveTransform(points1, points2)


# I1 = cv2.imread('../bg_2.png')
# I2 = cv2.imread('../field.jpg')
#
# output_size = (I2.shape[1], I2.shape[0])
# H = get_perspective_transform(2)
# J = cv2.warpPerspective(I1, H, output_size)
#
# cv2.imshow('I1', I1)
# cv2.waitKey(0)
#
# cv2.imshow('I2', I2)
# cv2.waitKey(0)
#
# cv2.imshow('J', J)
# cv2.waitKey(0)
