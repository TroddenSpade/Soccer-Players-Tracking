import cv2
import numpy as np

cap = cv2.VideoCapture('0_2_mini.mp4')
ret, I = cap.read()

avg = np.float32(I)

while ret:
    ret, I = cap.read()
    if not ret:
        break

    I = cv2.GaussianBlur(I,(5,5),0)

    cv2.accumulateWeighted(I,avg,0.004)
	
    bg = cv2.convertScaleAbs(avg)


cv2.imwrite('./bg_2.png', bg)

cv2.destroyAllWindows()
cap.release()

