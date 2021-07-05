import numpy as np
import cv2
import matplotlib.pyplot as plt

from transformation import get_perspective_transform

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80

file_name = '0_1'
cap = cv2.VideoCapture(file_name + ".mp4")

if (cap.isOpened()== False):
    print("Error opening video stream or file")

CUT_X = 70
H = get_perspective_transform(type=1)

frame_count = 0
img_count = 0
names = []

while True:
    frame_count += 1
    if frame_count > 2000:
        break
    ret, I_org = cap.read()
    if not ret:
        break
    
    ROI = I_org[:,:]
    FIELD = cv2.imread("field.jpg")

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

    pts = []
    cord = []
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

            pts.append([[x_p, y+h]])
            cord.append([(cX-l,y),(cX+l,y+h)])

    pts = np.array(pts, np.float32)
    if len(pts) > 0:
        out = cv2.perspectiveTransform(np.array(pts, np.float32), H).reshape(-1,2)
        for i, pt in enumerate(out):
            if pt[1] > 22:
                cv2.rectangle(ROI, cord[i][0], cord[i][1], (0,0,255), 2)
                cv2.circle(FIELD, (pt[0], pt[1]), 5, (0, 0, 255), -1) # center

                if frame_count % 20 == 0 & True: #frame_count > 1000;
                    name = "img/" + str(file_name) + "-" + str(frame_count) + "-" + str(img_count) + '.jpg'
                    cv2.imwrite(name, ROI[y:y+h, x:x+w])
                    names.append(name)
                    img_count += 1


    # output_size = (FIELD.shape[1], FIELD.shape[0])
    # J = cv2.warpPerspective(ROI, H, output_size)


    if ret == True:
        cv2.imshow('soccer mask', ROI)
        cv2.imshow('soccer Map', FIELD)
    else:
        break
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

names_arr = np.array(names)
np.save("names.npy", names_arr)

cap.release()
cv2.destroyAllWindows()