import numpy as np
import cv2 as cv

def draw_line(image, point1, point2, color=255):
    x0, y0 = point1
    x1, y1 = point2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return image

def convolute_image(image, kernel_image):
    kernel_sum = kernel_image.sum()
    kernel = kernel_image.astype(np.float32) / kernel_image.sum()
    conv = cv.filter2D(src=image, ddepth=-1, kernel=kernel)
    return conv

image = cv.imread('input.png', cv.IMREAD_GRAYSCALE)
cv.imwrite('grayscale.jpeg', image)
cv.imshow('grayscale', image)


line_image = np.zeros((500, 500), dtype='uint8')

point1 = (100, 100)
point2 = (400, 400)
line_image = draw_line(line_image, point1, point2)
cv.imwrite('line.jpeg', line_image)
cv.imshow('line', line_image)

conv = convolute_image(image, line_image)

cv.imshow('image', conv)
cv.imwrite('conclution.jpeg', conv)
cv.waitKey(0)
cv.destroyAllWindows()