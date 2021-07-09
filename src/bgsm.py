import numpy as np
import cv2
import matplotlib.pyplot as plt

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()

mask0 = np.load("masks/bg_0_mask.npy")
mask1 = np.load("masks/bg_1_mask.npy")
mask2 = np.load("masks/bg_2_mask.npy")

def bgs(I_org, type):

    if type == 0:
        ROI = cv2.bitwise_and(I_org, I_org, mask = mask0)
    if type == 1:
        ROI = cv2.bitwise_and(I_org, I_org, mask = mask1)
    if type == 2:
        ROI = cv2.bitwise_and(I_org, I_org, mask = mask2)

    I = cv2.GaussianBlur(ROI,(5,5),0)

    I_mog = mog2_backsub.apply(I)
    I_knn = knn_backsub.apply(I)

    mog_rate = 0.6
    I = ((1-mog_rate)*I_knn + mog_rate*I_mog).astype(np.uint8)

    _, I = cv2.threshold(I, 254, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((17, 7), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)

    # n_mog, C, stats, centroids = cv2.connectedComponentsWithStats(I);
    # for i in range(1,n_mog):
    #     if stats[i][4] < 60:7
    #         I[C == i] = 0

    return I


# cap0 = cv2.VideoCapture('videos/0_0.mp4')
# cap1 = cv2.VideoCapture('videos/0_1.mp4')
# cap2 = cv2.VideoCapture('videos/0_2.mp4')

# while True:
#     ret, I0 = cap0.read()
#     ret, I1 = cap1.read()
#     ret, I2 = cap2.read()
#     if not ret:
#         break
    
#     I0_bin = bgs(I0, type=0)
#     I1_bin = bgs(I1, type=1)
#     I2_bin = bgs(I2, type=2)


#     scale = 0.4
#     width = int(I1.shape[1] * scale)
#     height = int(I1.shape[0] * scale)
#     dim = (width, height)

#     I_bin = np.hstack((cv2.resize(I0_bin, dim), cv2.resize(I1_bin, dim), cv2.resize(I2_bin, dim)))
#     I = np.hstack((cv2.resize(I0, dim), cv2.resize(I1, dim), cv2.resize(I2, dim)))


#     if ret == True:
#         cv2.imshow('soccer', I)
#         cv2.imshow('soccer binary', I_bin)
        
#     else:
#         break

#     key = cv2.waitKey(60)
#     if key == ord('q'):
#         break

# cap0.release()
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()