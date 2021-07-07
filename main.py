import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.transformation import get_perspective_transform
from src.bgsm import bgs
from src.contour import get_contours
from src.classifier import classify, preprocess

mog2_backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) #60
knn_backsub = cv2.createBackgroundSubtractorKNN(detectShadows=True) #80
# backSub = cv2.bgsegm.BackgroundSubtractorGMG() #20
# mog_backsub = cv2.bgsegm.createBackgroundSubtractorMOG()

file_name = '0_1_mini'
cap = cv2.VideoCapture("./videos/" + file_name + ".mp4")

if (cap.isOpened()== False):
    print("Error opening video stream or file")

CUT_X = 70
H0 = get_perspective_transform(type=0)
H1 = get_perspective_transform(type=1)
H2 = get_perspective_transform(type=2)

frame_count = 0
img_count = 0
names = []

while True:
    frame_count += 1
    if frame_count > 400:
        break
    ret, I_org = cap.read()
    if not ret:
        break

    FIELD = cv2.imread("field.jpg")
    
    binary_img = bgs(I_org)

    feet, rec_bounds, bounds = get_contours(binary_img) 

    player_imgs = []
    circles = []
    pts = np.array(feet, np.float32)
    if len(pts) > 0:
        out = cv2.perspectiveTransform(np.array(pts, np.float32), H1).reshape(-1,2)
        for i, pt in enumerate(out):
            if pt[1] > 30:
                x,y, z,v = bounds[i]

                preporcc_img = preprocess(I_org[y:v, x:z])
                player_imgs.append(preporcc_img)
                # print("/////////////")
                # np.save("p.npy", preporcc_img)
                # cv2.imshow("x", preporcc_img)
                # cv2.waitKey()

                circles.append((pt[0], pt[1]))
                cv2.rectangle(I_org, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 2)

                ######################
                ###################### box saving
                # if frame_count % 15 == 0 & True: #frame_count > 1000;
                #     name = "img/" + str(file_name) + "-" + str(frame_count) + "-" + str(img_count) + '.jpg'
                #     cv2.imwrite(name, I_org[y:v, x:z])
                #     names.append(name)
                #     img_count += 1

    if len(player_imgs) > 0:
        res = classify(np.array(player_imgs))

    for i, pts in enumerate(circles):
        if res[i] == 0:
            cv2.circle(FIELD, pts, 5, (0, 0, 255), -1) # center
        elif res[i] == 1:
            cv2.circle(FIELD, pts, 5, (0, 255, 255), -1) # center
        elif res[i] == 2:
            cv2.circle(FIELD, pts, 5, (255, 0, 0), -1) # center


    if ret == True:
        cv2.imshow('soccer mask', I_org)
        cv2.imshow('soccer Map', FIELD)
    else:
        break

    key = cv2.waitKey(70)
    if key == ord('q'):
        break

# names_arr = np.array(names)
# np.save("names.npy", names_arr)

cap.release()
cv2.destroyAllWindows()