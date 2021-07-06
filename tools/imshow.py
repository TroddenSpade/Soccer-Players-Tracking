import numpy as np
import matplotlib.pyplot as plt
import cv2

I0 = cv2.imread("bg_0.png")
I1 = cv2.imread("bg_1.png")
I2 = cv2.imread("bg_2.png")

fig0,ax0 = plt.subplots()
ax0.imshow(I0)

fig1,ax1 = plt.subplots()
ax1.imshow(I1)

fig2,ax2 = plt.subplots()
ax2.imshow(I2)

plt.show()