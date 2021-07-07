import numpy as np
import cv2
import matplotlib.pyplot as plt

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()

def bgs(I_org):

    ROI = I_org[:,:]

    I = cv2.GaussianBlur(ROI,(5,5),0)

    I_mog = mog2_backsub.apply(I)
    I_knn = knn_backsub.apply(I)

    mog_rate = 0.8
    I = ((1-mog_rate)*I_knn + mog_rate*I_mog).astype(np.uint8)

    _, I = cv2.threshold(I, 254, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 7), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)

    # n_mog, C, stats, centroids = cv2.connectedComponentsWithStats(I);
    # for i in range(1,n_mog):
    #     if stats[i][4] < 60:7
    #         I[C == i] = 0

    return I


# cap = cv2.VideoCapture('output.mp4')
# if (cap.isOpened()== False):
#     print("Error opening video stream or file")

# while True:
#     ret, I_org = cap.read()
#     if not ret:
#         break
    
#     I = bgs(I_org)


#     if ret == True:
#         cv2.imshow('soccer test',I)
        
#     else:
#         break
#     key = cv2.waitKey(20)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()