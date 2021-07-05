import cv2
import numpy as np
import matplotlib.pyplot as plt

stitcher = cv2.createStitcher(False)

I0_org = cv2.imread('bg_0.png') # left
I1_org = cv2.imread('bg_1.png') # middle


x_t = 3000
y_t = 0

output_size = (I1_org.shape[1]+x_t, 1250)
M = np.float32([
	[1, 0, x_t],
	[0, 1, y_t]
])

I1 = cv2.warpAffine(I1_org, M, output_size)

def get_perspective_transform():
    n = 626
    m = 417

    points0 = np.array([(893, 191),
                        (1209, 197),
                        (1240, 309),
                        (736, 764)]).astype(np.float32) # left

    points1 = np.array([(136 + x_t, 168 + y_t),
                        (431 + x_t, 116 + y_t),
                        (492 + x_t, 204 + y_t),
                        (254 + x_t, 831 + y_t)]).astype(np.float32) # middle

    for i in range(4):
        cv2.circle(I0_org, (int(points0[i, 0]), int(points0[i, 1])), 3, [0, 0, 255], 2)
        cv2.circle(I1, (int(points1[i, 0]), int(points1[i, 1])), 3, [0, 0, 255], 2)

    # compute homography from point correspondences
    return cv2.getPerspectiveTransform(points0, points1)


H = get_perspective_transform()
I0 = cv2.warpPerspective(I0_org, H, (3400,1250))
# imgOutput = cv2.resize(imgOutput, (width,height))

result = stitcher.stitch((I0,I1_org))
print(result)

fig0,ax0 = plt.subplots()
ax0.imshow(I0_org)

fig1,ax1 = plt.subplots()
ax1.imshow(I1)

fig1,ax1 = plt.subplots()
ax1.imshow(result[1])

plt.show()