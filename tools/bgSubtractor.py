import numpy as np
import cv2
import matplotlib.pyplot as plt

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80

CUT_X = 70

cap = cv2.VideoCapture('output.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while True:
    ret, I_org = cap.read()
    if not ret:
        break
    

    ROI = I_org[CUT_X:,:]

    I = cv2.GaussianBlur(ROI,(3,3),0)

    I_mog = mog2_backsub.apply(I)
    I_knn = knn_backsub.apply(I)

    mog_rate = 0.8
    I = ((1-mog_rate)*I_knn + mog_rate*I_mog).astype(np.uint8)

    _, I = cv2.threshold(I, 254, 255, cv2.THRESH_BINARY)
    # _, I_mog = cv2.threshold(I_mog, 240, 255, cv2.THRESH_BINARY)
    # _, I_knn = cv2.threshold(I_knn, 240, 255, cv2.THRESH_BINARY)


    kernel = np.ones((5, 5), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((11, 1), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)
    # I = cv2.erode(I, kernel, iterations = 1)


    if ret == True:
        cv2.imshow('soccer mask',I)
        
    else:
        break
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()