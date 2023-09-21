import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cam = cv.VideoCapture(2)
cam.set(3, 800)
cam.set(4, 600)
img = None
while True:
    _, img = cam.read()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Img", img)
    if cv.waitKey(1) & 0xFF == ord('c'):
        cv.destroyAllWindows()
        break

edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
