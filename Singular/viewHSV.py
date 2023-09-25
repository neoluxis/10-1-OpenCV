import numpy as np
import cv2 as cv


def nothing(x):
    pass


def processImg(bgr, hsvl, hsvh):
    fom = cv.resize(bgr, (0, 0), fx=0.6, fy=0.6, interpolation=cv.INTER_NEAREST)
    hsv = cv.cvtColor(fom, cv.COLOR_BGR2HSV)
    tgt = cv.inRange(hsv, hsvl, hsvh)
    res = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow("Unprocessed Image", fom)
    cv.waitKey(1)
    cv.imshow("Processed Image", tgt)
    cv.waitKey(1)


def main(img=None):
    cv.namedWindow('HSV Bars')
    # 创建颜色变化的轨迹栏
    cv.createTrackbar('HL', 'HSV Bars', 0, 360, nothing)
    cv.createTrackbar('HH', 'HSV Bars', 0, 360, nothing)
    cv.createTrackbar('SL', 'HSV Bars', 0, 255, nothing)
    cv.createTrackbar('SH', 'HSV Bars', 0, 255, nothing)
    cv.createTrackbar('VL', 'HSV Bars', 0, 255, nothing)
    cv.createTrackbar('VH', 'HSV Bars', 0, 255, nothing)
    # 为 ON/OFF 功能创建开关
    switch = '0 : OFF \n1 : ON'
    cv.createTrackbar(switch, 'HSV Bars', 0, 1, nothing)

    cv.namedWindow("Unprocessed Image")
    cv.waitKey(1)
    cv.namedWindow("Processed Image")
    cv.waitKey(1)
    if type(img) == type(None):
        img = cv.imread('../assets/tasks/page06.png')
    while True:
        hsv_low = np.array([
            cv.getTrackbarPos("HL", "HSV Bars"),
            cv.getTrackbarPos("SL", "HSV Bars"),
            cv.getTrackbarPos("VL", "HSV Bars"),
        ])
        hsv_high = np.array([
            cv.getTrackbarPos("HH", "HSV Bars"),
            cv.getTrackbarPos("SH", "HSV Bars"),
            cv.getTrackbarPos("VH", "HSV Bars")
        ])
        s = cv.getTrackbarPos(switch, 'HSV Bars')
        # if s == 0:
        #     pass
        # else:
        #     processImg(img, hsv_low, hsv_high)
        processImg(img, hsv_low, hsv_high)
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
