import cv2
import numpy as np
import matplotlib.pyplot as plt

I0 = cv2.imread('bg_0.png') # left
I1 = cv2.imread('bg_1.png') # middle

output_size = (2 * I1.shape[1], 2 * I1.shape[0])

x_t = 0
y_t = 0
M = np.float32([
	[1, 0, x_t],
	[0, 1, y_t]
])

I1 = cv2.warpAffine(I1, M, output_size)

def get_perspective_transform():
    n = 626
    m = 417

    points0 = np.array([(893, 191),
                        (1209, 197),
                        (1240, 309),
                        (743, 769)]).astype(np.float32) # left

    points1 = np.array([(136 + x_t, 168 + y_t),
                        (431 + x_t, 116 + y_t),
                        (492 + x_t, 204 + y_t),
                        (254 + x_t, 831 + y_t)]).astype(np.float32) # middle

    for i in range(4):
        cv2.circle(I0, (int(points0[i, 0]), int(points0[i, 1])), 3, [0, 0, 255], 2)
        cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)

    # compute homography from point correspondences
    return cv2.getPerspectiveTransform(points0, points1)


H = get_perspective_transform()
J = cv2.warpPerspective(I0, H, output_size)
# imgOutput = cv2.resize(imgOutput, (width,height))


fig0,ax0 = plt.subplots()
ax0.imshow(I0)

fig1,ax1 = plt.subplots()
ax1.imshow(I1)

fig2,ax2 = plt.subplots()
ax2.imshow(J)

plt.show()