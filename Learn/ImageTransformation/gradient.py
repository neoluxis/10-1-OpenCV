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
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
