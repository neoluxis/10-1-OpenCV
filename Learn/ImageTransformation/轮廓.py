import cv2 as  cv
import numpy as np

if __name__ == "__main__":
    cam = cv.VideoCapture(2)
    cam.set(3, 800)
    cam.set(4, 600)
    # 800x800 image
    img = np.zeros((800, 800, 3), np.uint8)
    # draw a while rectangle in center
    cv.rectangle(img, (300, 300), (500, 500), (255, 255, 255), 1)
    # fill the rectangle with white
    img[300:500, 300:500] = [255, 255, 255]
    while True:
        # _, img = cam.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(img, 0, 5, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img, contours, 0, (0, 255, 0), 1)
        cv.imshow("Image", img)
        if cv.waitKey(0) & 0xff == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()
