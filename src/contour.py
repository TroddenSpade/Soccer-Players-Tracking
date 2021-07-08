import numpy as np
import cv2

def get_contours(bin_image):
    feet = []
    rec_bounds = []
    bounds = []
    areas = []
    contours, _ = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (area > 70) and (w<2.3*h):            
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])

            dis = x+w//2 - cX 
            x_p = cX - dis
            l = min(cX-x, x+w-cX)

            feet.append([[x_p, y+h]])
            rec_bounds.append([(cX-l,y),(cX+l,y+h)])
            bounds.append((x,y,x+w,y+h))
            areas.append(area)

    return feet, rec_bounds, bounds, areas