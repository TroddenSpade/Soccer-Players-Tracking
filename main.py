import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.transformation import get_perspective_transform
from src.bgsm import bgs
from src.contour import get_contours
from src.classifier import classify, preprocess


cap0 = cv2.VideoCapture("./videos/" + '0_0' + ".mp4")
cap1 = cv2.VideoCapture("./videos/" + '0_1' + ".mp4")
cap2 = cv2.VideoCapture("./videos/" + '0_2' + ".mp4")

H0 = get_perspective_transform(type=0)
H1 = get_perspective_transform(type=1)
H2 = get_perspective_transform(type=2)

left_mask = np.load("masks/I_left.npy")
center_mask = np.load("masks/I_center.npy")
right_mask = np.load("masks/I_right.npy")

frame_count = 0
img_count = 0
names = []

while True:
    frame_count += 1
    ret0, I0 = cap0.read()
    ret1, I1 = cap1.read()
    ret2, I2 = cap2.read()
    if not ret1 and not ret0 and not ret2:
        break

    FIELD = cv2.imread("field.jpg")
    player_imgs = []
    circles = []

    
    binary_img0 = bgs(I0, type=0)
    feet, rec_bounds, bounds, areas = get_contours(binary_img0) 
    ## Camera 0
    pts = np.array(feet, np.float32)
    if len(pts) > 0:
        out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H0).reshape(-1,2)
        for i, pt in enumerate(out1):
            # filter based on area and position
            if left_mask[int(pt[1]), int(pt[0])]: # and pt[1]*pt[1]/150 < areas[i]:
                x,y, z,v = bounds[i]

                preporcc_img = preprocess(I0[y:v, x:z])
                player_imgs.append(preporcc_img)
                circles.append((pt[0], pt[1]))
                cv2.rectangle(I0, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)

    binary_img1 = bgs(I1, type=1)
    feet, rec_bounds, bounds, areas = get_contours(binary_img1) 
    ## Camera 1
    pts = np.array(feet, np.float32)
    if len(pts) > 0:
        out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H1).reshape(-1,2)
        for i, pt in enumerate(out1):
            # filter based on area and position
            if center_mask[int(pt[1]), int(pt[0])] and pt[1]*pt[1]/170 < areas[i]:
                x,y, z,v = bounds[i]

                preporcc_img = preprocess(I1[y:v, x:z])
                player_imgs.append(preporcc_img)
                circles.append((pt[0], pt[1]))
                cv2.rectangle(I1, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)

    binary_img2 = bgs(I2, type=2)
    feet, rec_bounds, bounds, areas = get_contours(binary_img2) 
    ## Camera 0
    pts = np.array(feet, np.float32)
    if len(pts) > 0:
        out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H2).reshape(-1,2)
        for i, pt in enumerate(out1):
            # filter based on area and position
            if right_mask[int(pt[1]), int(pt[0])]: # and pt[1]*pt[1]/150 < areas[i]:
                x,y, z,v = bounds[i]

                preporcc_img = preprocess(I2[y:v, x:z])
                player_imgs.append(preporcc_img)
                circles.append((pt[0], pt[1]))
                cv2.rectangle(I2, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)

    if len(player_imgs) > 0:
        res = classify(np.array(player_imgs))

    for i, point in enumerate(circles):
        if res[i] == 0:
            cv2.circle(FIELD, point, 7, (255, 0, 0), -1)
        elif res[i] == 1:
            cv2.circle(FIELD, point, 7, (0, 255, 255), -1)
        elif res[i] == 2:
            cv2.circle(FIELD, point, 7, (0, 0, 255), -1)


    scale = 0.4
    width = int(I1.shape[1] * scale)
    height = int(I1.shape[0] * scale)
    dim = (width, height)
    
    I = np.hstack((cv2.resize(I0, dim), cv2.resize(I1, dim), cv2.resize(I2, dim)))

    if ret1 == True and ret0 == True and ret2 == True:
        cv2.imshow('soccer mask', I)
        cv2.imshow('soccer Map', FIELD)
    else:
        break

    key = cv2.waitKey(70)
    if key == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()