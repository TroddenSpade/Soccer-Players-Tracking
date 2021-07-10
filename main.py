import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.transformation import get_perspective_transform
from src.bgsm import bgs
from src.contour import get_contours
from src.classifier import classify, preprocess


cap0 = cv2.VideoCapture("./videos/" + '1_0' + ".mp4")
cap1 = cv2.VideoCapture("./videos/" + '1_1' + ".mp4")
cap2 = cv2.VideoCapture("./videos/" + '1_2' + ".mp4")

H0 = get_perspective_transform(type=0)
H1 = get_perspective_transform(type=1)
H2 = get_perspective_transform(type=2)

left_mask = np.load("masks/I_left.npy")
center_mask = np.load("masks/I_center.npy")
right_mask = np.load("masks/I_right.npy")

tracker_method = cv2.legacy.TrackerKCF_create

trackers1 = cv2.legacy.MultiTracker_create()

SKIP_FRAME = 5
frame_count = 0
img_count = 0
names = []

while True:
    frame_count += 1
    # ret0, I0 = cap0.read()
    ret1, I1 = cap1.read()
    # ret2, I2 = cap2.read()
    # if not ret1 and not ret0 and not ret2:
    #     break

    FIELD = cv2.imread("field.jpg")
    player_imgs = []
    circles = []

    # output_size = (FIELD.shape[1], FIELD.shape[0])
    # J0 = cv2.warpPerspective(I0, H0, output_size)
    # J1 = cv2.warpPerspective(I1, H1, output_size)
    # J2 = cv2.warpPerspective(I2, H2, output_size)
    # mask0 = np.load("masks/I_left.npy").astype(np.uint8)
    # mask1 = np.load("masks/I_center.npy").astype(np.uint8)
    # mask2 = np.load("masks/I_right.npy").astype(np.uint8)
    # ROI0 = cv2.bitwise_and(J0, J0, mask = mask0)
    # ROI1 = cv2.bitwise_and(J1, J1, mask = mask1)
    # ROI2 = cv2.bitwise_and(J2, J2, mask = mask2)
    # scale = 0.7
    # width = int(FIELD.shape[1] * scale)
    # height = int(FIELD.shape[0] * scale)
    # dim = (width, height)

    
    # binary_img0 = bgs(I0, type=0)
    # feet, rec_bounds, bounds, areas = get_contours(binary_img0) 
    # ## Camera 0
    # pts = np.array(feet, np.float32)
    # if len(pts) > 0:
    #     out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H0).reshape(-1,2)
    #     for i, pt in enumerate(out1):
    #         # filter based on area and position
    #         if left_mask[int(pt[1]), int(pt[0])] and pt[1]**2/300 < areas[i]:
    #             x,y, z,v = bounds[i]

    #             preporcc_img = preprocess(I0[y:v, x:z])
    #             player_imgs.append(preporcc_img)
    #             circles.append((pt[0], pt[1]))
    #             cv2.rectangle(I0, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
    #             # cv2.circle(ROI0, (pt[0], pt[1]), 5, (0, 255, 255), -1)


    binary_img1 = bgs(I1, type=1)
    (success, boxes) = trackers1.update(I1)
    if frame_count % SKIP_FRAME == 0: 
        trackers1 = cv2.legacy.MultiTracker_create()
        feet, rec_bounds, boxes, areas = get_contours(binary_img1) 
        ## Camera 1
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H1).reshape(-1,2)
            for i, pt in enumerate(out1):
                if center_mask[int(pt[1]), int(pt[0])] and pt[1]**2/240 < areas[i]:
                    x, y, w, h = boxes[i]
                    tracker = tracker_method()
                    trackers1.add(tracker, I1, boxes[i])
                    preporcc_img = preprocess(I1[y:y+h, x:x+w])
                    player_imgs.append(preporcc_img)
                    circles.append((pt[0], pt[1]))
                    # cv2.rectangle(I1, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
                    # cv2.circle(ROI1, (pt[0], pt[1]), 5, (0, 255, 255), -1)
    else:
        feet = []
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(I1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            feet.append([[x + w//2, y+h]])
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out1 = cv2.perspectiveTransform(np.array(pts), H1).reshape(-1,2)
            for i, pt in enumerate(out1):
                circles.append((pt[0], pt[1]))


    # binary_img2 = bgs(I2, type=2)
    # feet, rec_bounds, bounds, areas = get_contours(binary_img2) 
    # ## Camera 2
    # pts = np.array(feet, np.float32)
    # if len(pts) > 0:
    #     out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H2).reshape(-1,2)
    #     for i, pt in enumerate(out1):
    #         # filter based on area and position
    #         x,y, z,v = bounds[i]
    #         if right_mask[int(pt[1]), int(pt[0])] and I2[y:v, x:z].mean() > 37: # and pt[1]*pt[1]/150 < areas[i]:

    #             preporcc_img = preprocess(I2[y:v, x:z])
    #             player_imgs.append(preporcc_img)
    #             circles.append((pt[0], pt[1]))
    #             cv2.rectangle(I2, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
    #             # cv2.circle(ROI2, (pt[0], pt[1]), 5, (0, 255, 255), -1)
    #             # cv2.putText(I2, str(I2[y:v, x:z].mean()), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    if frame_count % SKIP_FRAME == 0 and len(player_imgs) > 0:
        res = classify(np.array(player_imgs))

    for i, point in enumerate(circles):
        if res[i] == 0: # blue
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (255, 0, 0), -1)
        elif res[i] == 1: # yellow
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
        elif res[i] == 2: # red
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (0, 0, 255), -1)


    # scale = 0.4
    # width = int(I1.shape[1] * scale)
    # height = int(I1.shape[0] * scale)
    # dim = (width, height)
    
    # I = np.hstack((cv2.resize(I0, dim), cv2.resize(I1, dim), cv2.resize(I2, dim)))


    # J = np.hstack((cv2.resize(ROI0, dim), cv2.resize(ROI1, dim), cv2.resize(ROI2, dim)))
    # cv2.imshow('J2', J)


    cv2.imshow('soccer mask', I1)
    cv2.imshow('soccer Map', FIELD)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()