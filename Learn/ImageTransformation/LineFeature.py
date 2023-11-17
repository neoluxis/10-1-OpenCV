# encoding:utf-8
import cv2
import numpy as np

filename = '../../assets/tasks/page06.png'

img = cv2.imread(filename)
img = cv2.resize(img, (640, 480))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 输入图像必须是float32， 最后一个参数[0.04,0.06]
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
cv2.imshow('dst', dst)
dst = cv2.dilate(dst, None)

img[dst >0.9 * dst.max()] = [0, 0, 255]
cv2.imshow('img', img)
cv2.imshow('dst2', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
