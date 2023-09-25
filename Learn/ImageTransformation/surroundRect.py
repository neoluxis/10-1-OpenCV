import cv2 as cv
import numpy as np

img = np.zeros((800, 800, 3), np.uint8)
# draw a while rectangle in center
# cv.rectangle(img, (300, 300), (500, 500), (255, 255, 255), 1)
# fill the rectangle with white
for i in range(0, 100):
    for j in range(int(350 - 0.4*i), int(350 + 0.6*i)):
        img[i + 300, j] = [255, 255, 255]

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img, 100, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
epsilon = 0.5*cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img, cnt, -1, (0, 255, 0), 2)
# x, y, w, h = cv.boundingRect(cnt)
# cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# rect = cv.minAreaRect(cnt)
# box = cv.boxPoints(rect)
# box = np.int0(box)
# cv.drawContours(img, [box], 0, (0, 0, 255), 2)
cv.imshow("Image", img)
cv.waitKey(0)
