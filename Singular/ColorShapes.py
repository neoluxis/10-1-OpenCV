import cv2 as cv
import numpy as np

import viewHSV as nl

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
}


def find(bgr):
    restr = ''
    strs = {
        'tri': "三角形: ",
        'rec': "矩形: ",
        'cir': "圆形: ",
        'red': "红色",
        'yellow': "黄色",
        'green': "绿色",
        'blue': "蓝色",
    }
    cv.imshow("bgr", bgr)
    cv.waitKey(1)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    for color in colors.keys():
        blb = cv.inRange(hsv, colors[color]['low'], colors[color]['high'])
        blb = cv.erode(blb, None, iterations=2)
        # cv.imshow("blb", blb)
        cv.waitKey(1)
        cnts, _ = cv.findContours(blb.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0]
        # print(len(cnts))
        for c in cnts:
            area = cv.contourArea(c)
            # if area < 100:
            #     continue
            # cv.drawContours(bgr, [c], -1, (0, 255, 0), 2)
            # cv.imshow("bgr", bgr)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # print(area)
            # print(cv.arcLength(c, True))
            # print(cv.approxPolyDP(c, 0.04 * cv.arcLength(c, True), True))
            shape = strs['cir']
            approx = cv.approxPolyDP(c, 0.04 * cv.arcLength(c, True), True)
            if len(approx) == 3:
                shape = strs['tri']
            if len(approx) == 4:
                shape = strs['rec']
            cv.drawContours(bgr, [c], -1, (0, 255, 0), 2)
            cv.imshow("bgr", bgr)
            cv.waitKey(1)
            restr += shape + strs[color] + '; '
            restr += "面积: " + str(area) + '; '
            restr += f"X: {c[0][0][0]}; Y: {c[0][0][1]};\n"
            # print(shape + strs[color])
    cv.waitKey(0)
    return restr


if __name__ == "__main__":
    ps = [
        cv.imread('../assets/tasks/page02.png'),
        cv.imread('../assets/tasks/page03.png'),
        cv.imread('../assets/tasks/page04.png'),
    ]

    for p in ps:
        print(find(p))

