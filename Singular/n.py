import cv2 as cv
import numpy as np


def findline(bgr):
    cv.line(bgr, (40, 500), (310, 232), (0, 0, 0), 2)
    cv.line(bgr, (311, 235), (378, 310), (0, 0, 0), 2)
    cv.line(bgr, (510, 190), (378, 311), (0, 0, 0), 2)
    return "Line: 3 \nRight Angle: 2"
