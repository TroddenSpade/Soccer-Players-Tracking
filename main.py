import numpy as np
import cv2
import matplotlib.pyplot as plt

from transformation import get_perspective_transform

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80

CUT_X = 70

cap = cv2.VideoCapture('output.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")

frame_count = 0
img_count = 0
names = []

while True:
    frame_count += 1
    ret, I_org = cap.read()
    if not ret:
        break
    
    ROI = I_org[CUT_X:,:]

    I = cv2.GaussianBlur(ROI,(5,5),0)

    I_mog = mog2_backsub.apply(I)
    I_knn = knn_backsub.apply(I)

    mog_rate = 0.8
    I = ((1-mog_rate)*I_knn + mog_rate*I_mog).astype(np.uint8)

    _, I = cv2.threshold(I, 254, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((11, 1), np.uint8)
    I = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)

    # n_mog, C, stats, centroids = cv2.connectedComponentsWithStats(I);
    # for i in range(1,n_mog):
    #     if stats[i][4] < 60:
    #         I[C == i] = 0

    _, contours, _ = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (area > 1200 * y/960) and (w<2.3*h):            
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            dis = x+w//2 - cX 
            x_p = cX - dis

            l = min(cX-x, x+w-cX)

            # if frame_count % 20 == 0:
            #     name = "img/" + str(img_count) + '.jpg'
            #     cv2.imwrite(name, ROI[y:y+h, x:x+w])
            #     names.append(name)
            #     img_count += 1

            cv2.rectangle(ROI, (cX-l,y), (cX+l,y+h), (0,0,255), 2)
            cv2.circle(ROI, (cX, cY), 5, (0, 0, 255), -1) # center
            cv2.circle(ROI, (x_p,y+h), 5, (0, 255, 0), -1) # foot
            cv2.putText(ROI, str() + "-" + str(y), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


    if ret == True:
        cv2.imshow('soccer mask',ROI)
    else:
        break
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

# names_arr = np.array(names)
# np.save("names.npy", names_arr)

cap.release()
cv2.destroyAllWindows()