import cv2 as cv
import numpy as np

if __name__ == "__main__":
    qrc = cv.imread('../assets/qrcode.png')
    barc = cv.imread('../assets/barcode.png')

    scanner = cv.QRCodeDetector()
    data, bbox, straight_qrcode = scanner.detectAndDecode(qrc)
    cv.imshow('QRCode', qrc)
    print(data)
    data, bbox, straight_qrcode = scanner.detectAndDecode(barc)
    cv.imshow('BarCode', barc)
    print(data if data else 'No Data')
    cv.waitKey(0)
