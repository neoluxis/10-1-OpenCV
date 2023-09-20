import cv2 as cv

if __name__ == "__main__":
    img = cv.imread('../blackarch-dm.png', 0)
    rows, cols = img.shape
    # cols-1 和 rows-1 是坐标限制
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 30, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
