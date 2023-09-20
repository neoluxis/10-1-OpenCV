import cv2 as cv
import numpy as np

if __name__ == "__main__":
    # take a photo
    cam = cv.VideoCapture(-1)
    _, img = cam.read()
    cv.imshow("Img", img)
    mask = cv.inRange(img, np.array([0, 0, 0]), np.array([50, 50, 50]))
    cv.imshow("Mask'", mask)
    res = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("Res", res)
    # cv.imshow("img", img)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("Gray", gray)
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower = np.array([110, 50, 50])
    # upper = np.array([130, 255, 255])
    # mask = cv.inRange(hsv, lower, upper)
    # res = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow("mask", mask)
    # cv.imshow("res", res)
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
