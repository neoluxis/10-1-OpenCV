import cv2 as cv


def s1(): # 1.1
    print("hello_")


def main(tsk=s1, img=None):
    img = img
    if not img:
        tsk()


if __name__ == "__main__":
    main()
