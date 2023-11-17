import math
import time

import viewHSV

import cv2 as cv
import numpy as np

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

    def rmdupseg(line_segments, distance_threshold=10, angle_threshold=np.pi / 180):
        """
        Remove duplicate line segments from an array of line segments.

        Parameters:
        - line_segments (numpy.ndarray): Array of line segments, each containing [x1, y1, x2, y2].
        - distance_threshold (int): Distance threshold for considering line segments as duplicates (default is 10).
        - angle_threshold (float): Angle threshold in radians for considering line segments as duplicates (default is 1 degree in radians).

        Returns:
        - numpy.ndarray: Array of unique line segments, each containing [x1, y1, x2, y2].
        """

        def linedist(line1, line2):
            """
            Calculate distance of 2 lines
            :param line1:
            :param line2:
            :return:
            """
            x11, y11, x12, y12 = line1[0]
            x21, y21, x22, y22 = line2[0]

        unique_line_segments = []  # Store unique line segments
        for segment in line_segments:
            isUnique = True
            x1, y1, x2, y2 = segment[0]
            dirvec1 = np.array([x2 - x1, y2 - y1])  # Direction vector of line segment
            for unique_segment in unique_line_segments:
                x3, y3, x4, y4 = unique_segment[0]
                dirvec2 = np.array([x4 - x3, y4 - y3])
                dot = np.dot(dirvec1, dirvec2)  # Dot product of direction vectors
                mag1 = np.linalg.norm(dirvec1)  # Magnitude of direction vectors
                mag2 = np.linalg.norm(dirvec2)
                angle = np.arccos(dot / (mag1 * mag2))  # Angle between line segments
                if angle < angle_threshold:  # If angle is less than threshold, check distance
                    isUnique = False
                else:
                    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))  # Distance between line segments
                    if dist > distance_threshold:  # If distance is less than threshold, line segment is a duplicate
                        isUnique = False
            if isUnique:
                unique_line_segments.append(segment)
        return np.array(unique_line_segments)

    def rmindex(arr, index):
        newarr = []
        for i in range(len(arr)):
            if i != index:
                newarr.append(arr[i])
        return np.array(newarr)

    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    blc = cv.inRange(hsv, colors['black']['low'], colors['black']['high'])
    x4, y4 = 219, 790
    blc = cv.erode(blc, None, iterations=6)
    blc = cv.dilate(blc, None, iterations=6)
    bcnt = cv.findContours(blc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    wimg = np.zeros(bgr.shape, np.uint8)
    x3, y3 = 275, 532
    for cnt in bcnt:
        cir = cv.minEnclosingCircle(cnt)
        cv.circle(wimg, (int(cir[0][0]), int(cir[0][1])), int(cir[1]) + 3, (255, 255, 255), -1)
        wimg = cv.bitwise_or(wimg, bgr)
    wimg = cv.cvtColor(wimg, cv.COLOR_BGR2HSV)
    blc = cv.inRange(wimg, colors['black']['low'], colors['black']['high'])
    blc = cv.Canny(blc, 50, 100, apertureSize=3)
    # blc = cv.dilate(blc, None, iterations=1)
    # blc = cv.erode(blc, np.ones((4, 2), np.uint8))
    lines = cv.HoughLinesP(blc, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=10)
    n = 0
    # lines = rmdupseg(lines, 1, np.pi / 180)
    lines = rmindex(lines, 9)
    for line in lines:
        n += 1
        x1, y1, x2, y2 = line[0]
        cv.line(bgr, (x1, y1), (x2, y2), (127, 127, 0), 2)
        cv.putText(bgr, f"l{n}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv.putText(bgr, f"{n}", (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    print(len(lines))
    cv.putText(bgr, "Crossing", (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv.putText(bgr, "Right Angle", (x4, y4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # cv.imshow("BLC", blc)
    cv.imshow("Blacks", bgr)


def blueLine(bgr):
    def distance(pos1, pos2):
        d = math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[2]) ** 2)
        return d

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    reds = cv.inRange(hsv, colors['red']['low'], colors['red']['high'])
    reds = cv.morphologyEx(reds, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # cv.imshow("reds", reds)
    contours, hierarchy = cv.findContours(reds, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # print(contours)
    ends = [[], []]
    for n in range(len(contours)):
        cnt = contours[n]
        (x, y), r = cv.minEnclosingCircle(cnt)
        # cv.circle(bgr, (int(x), int(y)), int(r), (0, 0, 0), 2)
        ends[n] = [int(x), int(y)]
    # cv.drawContours(bgr, contours, -1, (0, 0, 0), 2)
    gray = np.float32(gray)
    cv.putText(bgr, "B", ends[0] if ends[0][0] < ends[1][0] else ends[1],  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv.putText(bgr, "A", ends[0] if ends[0][0] > ends[1][0] else ends[1],  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # 输入图像必须是float32， 最后一个参数[0.04,0.06]
    dst = cv.cornerHarris(gray, 2, 3, 0.06)
    # cv.imshow('dst', dst)
    dst = cv.dilate(dst, None)

    gr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    gr[dst > 0.9 * dst.max()] = [0, 0, 255]
    # cv.imshow("gr", gr)
    # print(dst > 0.9 * dst.max())
    cv.imshow("bgr", bgr)


if __name__ == "__main__":
    p5 = cv.imread("../assets/tasks/page05.png")
    p6 = cv.imread("../assets/tasks/page06.png")
    while True:
        # greenLineCircle(p5)
        # time.sleep(1)
        # blacklines(p5)
        # time.sleep(1)
        blueLine(p6)
        if cv.waitKey(0) & 0xff == ord('q'):
            break
    cv.destroyAllWindows()
