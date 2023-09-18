from ParseInp import Parser
import cv2 as cv

tasklist = []

def Kam():
    """
    Take a photo with camera

    :return: A Mat object of the image
    """
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return None
    stts, img = cam.read()
    if not stts:
        print("Cannot read camera")
        return None
    cam.release()
    return img


def parsen(arr, useCamera=False):
    a, b = arr[-2], [-1]


def main(useCamera=False):
    parser = Parser()
    tsk = "None"
    while not parser.valid(tsk)[0]:
        tsk = input("Task Number(x.x): ")

    tsk = parser.valid(tsk)[1:]
    # print(tsk)
    parsen(tsk, useCamera)


if __name__ == "__main__":
    main()
