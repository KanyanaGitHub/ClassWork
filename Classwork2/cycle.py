import cv2 as cv
import numpy as np


image = cv.imread("input.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)


contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
center = np.zeros(thresh.shape, dtype='uint8')
cv.drawContours(center, contours, -1, (255, 0, 0), 1)


center = cv.copyMakeBorder(center, 100, 100, 100, 100, cv.BORDER_CONSTANT, None, value=0)


circle_stamp = np.zeros((100, 100), dtype="uint8")
circle_stamp = cv.circle(circle_stamp, (50, 50), 60, 10, 2)


img_h, img_w = center.shape
center_output = center.copy()

for y in range(0, img_h):
    for x in range(0, img_w):
        if center[y, x] > 200:
            
            if center_output[y - 50:y + 50, x - 50:x + 50].shape != (100, 100):
                break
            center_output[y - 50:y + 50, x - 50:x + 50] += circle_stamp[0:100, 0:100]


cv.imwrite("center.png", center_output)
cv.imshow("center_pic", center_output)
cv.waitKey(0)
cv.destroyAllWindows()