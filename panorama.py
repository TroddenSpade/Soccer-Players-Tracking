import cv2
import numpy as np
import matplotlib.pyplot as plt

stitcher = cv2.createStitcher(False)

I0_org = cv2.imread('bg_0.png') # left
I1_org = cv2.imread('bg_1.png') # middle

def get_perspective_transform(I0_org, I1_org):

    x_t = 3000
    y_t = 0

    output_size = (I1_org.shape[1]+x_t, 1250)
    M = np.float32([
        [1, 0, x_t],
        [0, 1, y_t]
    ])

    points0 = np.array([(893, 191),
                        (1209, 197),
                        (1240, 309),
                        (736, 764)]).astype(np.float32) # left

    points1 = np.array([(136 + x_t, 168 + y_t),
                        (431 + x_t, 116 + y_t),
                        (492 + x_t, 204 + y_t),
                        (254 + x_t, 831 + y_t)]).astype(np.float32) # middle

    # compute homography from point correspondences
    H = cv2.getPerspectiveTransform(points0, points1)

    I0 = cv2.warpPerspective(I0_org, H, (3400,1250))
    res, img = stitcher.stitch((I0,I1_org))
    width = int(img.shape[1] * 20 / 100)
    height = int(img.shape[0] * 20 / 100)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

# J = get_perspective_transform(I0_org, I1_org)
# cv2.imshow("j", J)
# cv2.waitKey()