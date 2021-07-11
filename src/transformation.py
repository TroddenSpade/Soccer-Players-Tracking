import cv2
import numpy as np


def get_perspective_transform(type):
    global points1, points2
    if type == 1:
        points1 = np.array([(1255, 744.7), #br
                            (836.3, 100.2), #tr
                            (432.2, 121.3), # tl
                            (230.35, 830)]).astype(np.float32) # bl

        points2 = np.array([(343.28, 371), #br
                            (408, 42.4), #tr
                            (223, 42.4), #tl
                            (267.9, 371)]).astype(np.float32) # bl

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


# I0 = cv2.imread('bg_0.png')
# I1 = cv2.imread('bg_1.png')
# I2 = cv2.imread('bg_2.png')
# I_field = cv2.imread('field.jpg')

# output_size = (I_field.shape[1], I_field.shape[0])
# H0 = get_perspective_transform(0)
# H1 = get_perspective_transform(1)
# H2 = get_perspective_transform(2)
# J0 = cv2.warpPerspective(I0, H0, output_size)
# J1 = cv2.warpPerspective(I1, H1, output_size)
# J2 = cv2.warpPerspective(I2, H2, output_size)


# mask0 = np.load("masks/I_left.npy").astype(np.uint8)
# mask1 = np.load("masks/I_center.npy").astype(np.uint8)
# mask2 = np.load("masks/I_right.npy").astype(np.uint8)

# ROI0 = cv2.bitwise_and(J0, J0, mask = mask0)
# ROI1 = cv2.bitwise_and(J1, J1, mask = mask1)
# ROI2 = cv2.bitwise_and(J2, J2, mask = mask2)

# scale = 1
# width = int(I_field.shape[1] * scale)
# height = int(I_field.shape[0] * scale)
# dim = (width, height)

# cv2.imshow('F', I_field)
# cv2.waitKey(0)

# I = np.hstack((cv2.resize(I0, dim), cv2.resize(I1, dim), cv2.resize(I2, dim)))
# cv2.imshow('I', I)
# cv2.waitKey(0)

# cv2.imshow('J', J1)
# cv2.waitKey(0)

# J = np.hstack((cv2.resize(ROI0, dim), cv2.resize(ROI1, dim), cv2.resize(ROI2, dim)))
# cv2.imshow('J2', J)
# cv2.waitKey(0)

