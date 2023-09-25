import time

import viewHSV

import cv2 as cv
import numpy as np

colors = {
    'green': {
        'low': np.array([35, 43, 46]),
        'high': np.array([77, 255, 255])
    },
    'black': {
        'low': np.array([0, 0, 0]),
        'high': np.array([180, 255, 46])
    },
    'blue': {
        'low': np.array([100, 43, 46]),
        'high': np.array([124, 255, 255])
    },
}


def greenLineCircle(bgr):
    """
    识别绿色直线及顶部圆形，图像上画出直线、色块并在终
端输出直线 X、Y 偏移坐标和角度。
    :param bgr:
    :return:
    """

    def averageLine(lines):
        count = 0
        x1, y1, x2, y2 = 0, 0, 0, 0
        for line in lines:
            count += 1
            xt1, yt1, xt2, yt2 = line[0]
            x1 += xt1
            y1 += yt1
            x2 += xt2
            y2 += yt2
        return x1 / count, y1 / count, x2 / count, y2 / count

    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    glc = cv.inRange(hsv, colors['green']['low'], colors['green']['high'])
    glc = cv.dilate(cv.erode(glc, None, iterations=5), None, iterations=4)
    gcnt = cv.findContours(glc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    (x, y), r = cv.minEnclosingCircle(gcnt[0])
    whiteimg = np.zeros(bgr.shape, np.uint8)
    whiteimg[:, :] = [255, 255, 255]
    cv.circle(bgr, (int(x), int(y)), int(r), (0, 0, 0), 2)
    cv.circle(whiteimg, (int(x), int(y)), int(r) + 5, (0, 0, 0), -1)
    whiteimg = cv.bitwise_and(whiteimg, bgr)
    hsv = cv.cvtColor(whiteimg, cv.COLOR_BGR2HSV)
    glc = cv.inRange(hsv, colors['green']['low'], colors['green']['high'])
    glc = cv.erode(glc, np.ones((5, 7), np.uint8))
    cv.imshow("Green Line Circle", glc)
    lines = cv.HoughLinesP(glc, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    # x1, y1, x2, y2 = averageLine(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv.imshow("Image", bgr)


def blacklines(bgr):
    """
    识别黑色直线顶部圆形并判断类型（纯直线、直角或十
字），在图像上画出直线、色块。

    :param bgr:
    :return:
    """
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    blc = cv.inRange(hsv, colors['black']['low'], colors['black']['high'])
    blc = cv.erode(blc, None, iterations=6)
    blc = cv.dilate(blc, None, iterations=6)
    bcnt = cv.findContours(blc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    wimg = np.zeros(bgr.shape, np.uint8)
    for cnt in bcnt:
        cir = cv.minEnclosingCircle(cnt)
        cv.circle(wimg, (int(cir[0][0]), int(cir[0][1])), int(cir[1]) + 3, (255, 255, 255), -1)
        wimg = cv.bitwise_or(wimg, bgr)
    wimg = cv.cvtColor(wimg, cv.COLOR_BGR2HSV)
    blc = cv.inRange(wimg, colors['black']['low'], colors['black']['high'])
    blc = cv.Canny(blc, 50, 100, apertureSize=3)
    # blc = cv.dilate(blc, None, iterations=1)
    # blc = cv.erode(blc, np.ones((4, 6), np.uint8))
    lines = cv.HoughLinesP(blc, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=10)
    n = 0
    for line in lines:
        n += 1
        x1, y1, x2, y2 = line[0]
        cv.line(bgr, (x1, y1), (x2, y2), (127, 127, 0), 1)
        cv.putText(bgr, f"{n}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print(len(lines))
    cv.imshow("BLC", blc)
    cv.imshow("Blacks", bgr)


if __name__ == "__main__":
    p5 = cv.imread("../assets/tasks/page05.png")
    while True:
        # greenLineCircle(p5)
        # time.sleep(1)
        blacklines(p5)
        if cv.waitKey(0) & 0xff == ord('q'):
            break
    cv.destroyAllWindows()
