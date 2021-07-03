import numpy as np
import cv2
import matplotlib.pyplot as plt

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()
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

    I = cv2.GaussianBlur(ROI,(5,5),0)

    I_mog = mog2_backsub.apply(I)
    I_knn = knn_backsub.apply(I)

    mog_rate = 0.6
    I = ((1-mog_rate)*I_knn + mog_rate*I_mog).astype(np.uint8)

    _, I = cv2.threshold(I, 250, 255, cv2.THRESH_BINARY)

    # n_mog, C, stats, centroids = cv2.connectedComponentsWithStats(I);
    # for i in range(1,n_mog):
    #     if stats[i][4] < 60:
    #         I[C == i] = 0

    kernel = np.ones((7, 7), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)

    _, contours, _ = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 800 * y/960:
            if w >= 3*h: continue
            # if w > h:
            #     w //=3
            #     x += 2 * w
            # cv2.drawContours(ROI, [cnt], -1, (0,255,0), 2)
            cv2.rectangle(ROI, (x,y), (x+w,y+h), (0,0,255), 2)



    if ret == True:
        cv2.imshow('soccer mask',ROI)

        
    else:
        break
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()