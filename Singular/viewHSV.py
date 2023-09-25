import numpy as np
import cv2 as cv


def nothing(x):
    pass


# 创建一个黑色的图像，一个窗口
img = np.zeros((300, 512, 3), np.uint8)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.namedWindow('image')
# 创建颜色变化的轨迹栏
cv.createTrackbar('H', 'image', 0, 360, nothing)
cv.createTrackbar('S', 'image', 0, 255, nothing)
cv.createTrackbar('V', 'image', 0, 255, nothing)
# 为 ON/OFF 功能创建开关
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)
while (1):
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到四条轨迹的当前位置
    r = cv.getTrackbarPos('H', 'image')
    g = cv.getTrackbarPos('S', 'image')
    b = cv.getTrackbarPos('V', 'image')
    s = cv.getTrackbarPos(switch, 'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]
cv.destroyAllWindows()
