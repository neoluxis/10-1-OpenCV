import cv2 as cv

if __name__ == "__main__":
    cam = cv.VideoCapture(2)
    cam.set(3, 640)
    cam.set(4, 320)
    while True:
        _, img = cam.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10)
        cv.imshow("Original", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()
