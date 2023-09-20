import cv2 as cv

if __name__ == "__main__":
    cam = cv.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        cv.imshow('frame', frame)
