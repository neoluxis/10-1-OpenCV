import time

import cv2 as cv
import numpy as np
import sys
import pandas as pd

# cam = cv.VideoCapture(1)
# cam.set(3, 800)
# cam.set(4, 600)

# HSV color space
colors = {
    'red': {
        'low': np.array([0, 60, 60]),
        'high': np.array([5, 255, 255])
    },
    'yellow': {
        'low': np.array([26, 43, 46]),
        'high': np.array([34, 255, 255])
    },
    'green': {
        'low': np.array([35, 43, 46]),
        'high': np.array([77, 255, 255])
    },
    'blue': {
        'low': np.array([100, 43, 46]),
        'high': np.array([124, 255, 255])
    },
    'white': {
        'low': np.array([0, 0, 221]),
        'high': np.array([180, 30, 255])
    },
    'black': {
        'low': np.array([0, 0, 0]),
        'high': np.array([180, 255, 46])
    },
}


def red_yellow(bgr):
    """
    分别识别追踪黄色、红色，终端输出颜色、坐标及对应面积。

    :param bgr:
    :return:
    """
    timg = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    rtgt = cv.inRange(timg, colors['red']['low'], colors['red']['high'])
    ytgt = cv.inRange(timg, colors['yellow']['low'], colors['yellow']['high'])
    rcnt = cv.findContours(rtgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ycnt = cv.findContours(ytgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ycnt = ycnt[0]
    rcnt = rcnt[0]
    timg = cv.cvtColor(timg, cv.COLOR_HSV2BGR)
    cv.drawContours(timg, rcnt, -1, (0, 0, 0), 2)
    cv.drawContours(timg, ycnt, -1, (0, 0, 0), 2)
    rM = cv.moments(rcnt[0])
    yM = cv.moments(ycnt[0])
    rX = int(rM['m10'] / rM['m00'])
    rY = int(rM['m01'] / rM['m00'])
    print(f"Red: Position: ({rX}, {rY}), Area: {rM['m00']}")
    yX = int(yM['m10'] / yM['m00'])
    yY = int(yM['m01'] / yM['m00'])
    print(f"Yellow: Position: ({yX}, {yY}), Area: {yM['m00']}")
    # cv.imshow("Red", rtgt)
    # cv.imshow("Yellow", ytgt)
    cv.imshow("1.1", timg)


def green_blue(bgr):
    """
     识别绿色和蓝色（注意滤除小块的蓝色），终端输出颜色、坐标及
对应面积。
    :param bgr:
    :return:
    """

    timg = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    gtgt = cv.inRange(timg, colors['green']['low'], colors['green']['high'])
    btgt = cv.inRange(timg, colors['blue']['low'], colors['blue']['high'])

    gtgt = cv.morphologyEx(gtgt, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    btgt = cv.morphologyEx(btgt, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    gcnt = cv.findContours(gtgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bcnt = cv.findContours(btgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    # gcnt = gcnt[0]
    # bcnt = bcnt[0]
    timg = cv.cvtColor(timg, cv.COLOR_HSV2BGR)
    cv.drawContours(timg, gcnt, -1, (0, 0, 0), 2)
    cv.drawContours(timg, bcnt, -1, (0, 0, 0), 2)
    gM = cv.moments(gcnt[0])
    bM = cv.moments(bcnt[0])
    gX = int(gM['m10'] / gM['m00'])
    gY = int(gM['m01'] / gM['m00'])
    print(f"Green: Position: ({gX}, {gY}), Area: {gM['m00']}")
    bX = int(bM['m10'] / bM['m00'])
    bY = int(bM['m01'] / bM['m00'])
    print(f"Blue: Position: ({bX}, {bY}), Area: {bM['m00']}")
    # cv.imshow("Green", gtgt)
    # cv.imshow("Blue", btgt)
    cv.imshow("1.2", timg)


def white_in_black(bgr):
    """
    识别黑色圆环内的白色块，终端输出颜色、坐标及对应面积。

    :param bgr:
    :return:
    """
    timg = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    btgt = cv.inRange(timg, colors['black']['low'], colors['black']['high'])
    # filt white in black
    btgt = cv.morphologyEx(btgt, cv.MORPH_CLOSE, np.ones((32, 32), np.uint8))
    btgt = cv.erode(btgt, np.ones((8, 8), np.uint8))
    bcnt = cv.findContours(btgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    ellipse = cv.fitEllipse(bcnt[0])
    blackimg = np.zeros(bgr.shape, np.uint8)
    mask = cv.ellipse(blackimg, ellipse, (255, 255, 255), -1)
    roi = cv.bitwise_and(bgr, mask)
    roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    wtgt = cv.inRange(roi, colors['white']['low'], colors['white']['high'])
    wcnt = cv.findContours(wtgt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    # cv.imshow("roi", roi)
    timg = cv.cvtColor(timg, cv.COLOR_HSV2BGR)
    cv.drawContours(timg, wcnt, -1, (0, 255, 0), 2)
    M = cv.moments(wcnt[0])
    X = int(M['m10'] / M['m00'])
    Y = int(M['m01'] / M['m00'])
    print(f"White: Position: ({X}, {Y}), Area: {M['m00']}")
    # cv.ellipse(timg, ellipse, (0, 0, 255), 2)
    # cv.drawContours(timg, bcnt, -1, (0, 255, 0), 2)
    # cv.imshow("Black", btgt)
    cv.imshow("1.3", timg)


if __name__ == "__main__":
    while True:
        # ret, img = cam.read()
        # if not ret:
        #     continue
        img = cv.imread("../assets/tasks/page07.png")
        red_yellow(img)
        cv.waitKey(10)
        time.sleep(1)
        green_blue(img)
        cv.waitKey(10)
        time.sleep(1)
        white_in_black(img)
        cv.waitKey(10)
        time.sleep(1)
        if cv.waitKey(0) & 0xff == ord('q'):
            break

    cv.destroyAllWindows()
    # cam.release()
