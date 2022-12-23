import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.transformation import get_perspective_transform
from src.bgsm import bgs
from src.contour import get_contours
from src.classifier import classify, preprocess


SKIP_FRAME = 8
TRACKING = False
TRANSFORMATIONS = True

FIELD = cv2.imread("field.jpg")
mask0 = np.load("masks/I_left.npy").astype(np.uint8)
mask1 = np.load("masks/I_center.npy").astype(np.uint8)
mask2 = np.load("masks/I_right.npy").astype(np.uint8)

scale = 0.8
width = int(FIELD.shape[1] * scale)
height = int(FIELD.shape[0] * scale)
dim = (width, height)

cap0 = cv2.VideoCapture("./videos/" + '0_0' + ".mp4")
cap1 = cv2.VideoCapture("./videos/" + '0_1' + ".mp4")
cap2 = cv2.VideoCapture("./videos/" + '0_2' + ".mp4")

H0 = get_perspective_transform(type=0)
H1 = get_perspective_transform(type=1)
H2 = get_perspective_transform(type=2)

left_mask = np.load("masks/I_left.npy")
center_mask = np.load("masks/I_center.npy")
right_mask = np.load("masks/I_right.npy")

tracker_method = cv2.legacy.TrackerCSRT_create

trackers0 = cv2.legacy.MultiTracker_create()
trackers1 = cv2.legacy.MultiTracker_create()
trackers2 = cv2.legacy.MultiTracker_create()

frame_count = 0
img_count = 0
names = []
res = []

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

    if TRANSFORMATIONS:
        output_size = (FIELD.shape[1], FIELD.shape[0])
        J0 = cv2.warpPerspective(I0, H0, output_size)
        J1 = cv2.warpPerspective(I1, H1, output_size)
        J2 = cv2.warpPerspective(I2, H2, output_size)
        ROI0 = cv2.bitwise_and(J0, J0, mask = mask0)
        ROI1 = cv2.bitwise_and(J1, J1, mask = mask1)
        ROI2 = cv2.bitwise_and(J2, J2, mask = mask2)

    # ## Camera 0
    binary_img0 = bgs(I0, type=0)
    if TRACKING: (success, boxes) = trackers0.update(I0)
    if frame_count % SKIP_FRAME == 0 or not TRACKING: 
        if TRACKING: trackers0 = cv2.legacy.MultiTracker_create()
        feet, rec_bounds, boxes, areas = get_contours(binary_img0)
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out0 = cv2.perspectiveTransform(np.array(pts, np.float32), H0).reshape(-1,2)
            for i, pt in enumerate(out0):
                if left_mask[int(pt[1]), int(pt[0])] and pt[1]**2/350 < areas[i]:
                    x, y, w, h = boxes[i]
                    if TRACKING:
                        tracker = tracker_method()
                        trackers0.add(tracker, I0, boxes[i])
                    preporcc_img = preprocess(I0[y:y+h, x:x+w])
                    player_imgs.append(preporcc_img)
                    circles.append((pt[0], pt[1]))
                    cv2.rectangle(I0, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
                    if TRANSFORMATIONS: cv2.circle(ROI0, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
    else:
        feet = []
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(I0, (x, y), (x + w, y + h), (0, 255, 0), 2)
            feet.append([[x + w//2, y+h]])
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out0 = cv2.perspectiveTransform(np.array(pts), H0).reshape(-1,2)
            for i, pt in enumerate(out0):
                circles.append((pt[0], pt[1]))

    # ## Camera 1
    binary_img1 = bgs(I1, type=1)
    (success, boxes) = trackers1.update(I1)
    if frame_count % SKIP_FRAME == 0 or not TRACKING: 
        if TRACKING: trackers1 = cv2.legacy.MultiTracker_create()
        feet, rec_bounds, boxes, areas = get_contours(binary_img1)
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out1 = cv2.perspectiveTransform(np.array(pts, np.float32), H1).reshape(-1,2)
            for i, pt in enumerate(out1):
                if center_mask[int(pt[1]), int(pt[0])] and pt[1]**2/260 < areas[i] and pt[1] > 32:
                    x, y, w, h = boxes[i]
                    if TRACKING:
                        tracker = tracker_method()
                        trackers1.add(tracker, I1, boxes[i])
                    preporcc_img = preprocess(I1[y:y+h, x:x+w])
                    player_imgs.append(preporcc_img)
                    circles.append((pt[0], pt[1]))
                    cv2.rectangle(I1, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
                    if TRANSFORMATIONS: cv2.circle(ROI1, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
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


    # ## Camera 2
    binary_img2 = bgs(I2, type=2)
    (success, boxes) = trackers2.update(I2)
    if frame_count % SKIP_FRAME == 0 or not TRACKING:
        if TRACKING: trackers2 = cv2.legacy.MultiTracker_create()
        feet, rec_bounds, boxes, areas = get_contours(binary_img2)
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out2 = cv2.perspectiveTransform(np.array(pts, np.float32), H2).reshape(-1,2)
            for i, pt in enumerate(out2):
                x, y, w, h = boxes[i]
                if right_mask[int(pt[1]), int(pt[0])] and I2[y:y+h, x:x+w].mean() > 30:                   
                    if TRACKING:
                        tracker = tracker_method()
                        trackers2.add(tracker, I2, boxes[i])
                    preporcc_img = preprocess(I2[y:y+h, x:x+w])
                    player_imgs.append(preporcc_img)
                    circles.append((pt[0], pt[1]))
                    cv2.rectangle(I2, rec_bounds[i][0], rec_bounds[i][1], (0,0,255), 1)
                    if TRANSFORMATIONS: cv2.circle(ROI2, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
    else:
        feet = []
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(I2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            feet.append([[x + w//2, y+h]])
        pts = np.array(feet, np.float32)
        if len(pts) > 0:
            out2 = cv2.perspectiveTransform(np.array(pts), H2).reshape(-1,2)
            for i, pt in enumerate(out2):
                circles.append((pt[0], pt[1]))

    if (not TRACKING or frame_count % SKIP_FRAME == 0) and len(player_imgs) > 0:
        res = classify(np.array(player_imgs))

    for i, point in enumerate(circles):
        if res[i] == 0: # blue
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (255, 0, 0), -1)
        elif res[i] == 1: # yellow
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
        elif res[i] == 2: # red
            cv2.circle(FIELD, (int(point[0]), int(point[1])), 7, (0, 0, 255), -1)
        if TRACKING:
            cv2.putText(FIELD, str(i), (int(point[0])-3, int(point[1])+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    
    I = np.hstack((cv2.resize(binary_img0, dim), cv2.resize(binary_img1, dim), cv2.resize(binary_img2, dim)))

    if TRANSFORMATIONS:
        J = np.hstack((cv2.resize(ROI0, dim), cv2.resize(ROI1, dim), cv2.resize(ROI2, dim)))
        cv2.imshow('Transformations', J)


    cv2.imshow('Soccer Field', I)
    cv2.imshow('Soccer Map', FIELD)

    key = cv2.waitKey(70)
    if key == ord('q'):
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()