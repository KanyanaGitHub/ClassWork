import cv2 as cv
import random
import numpy as np

img = cv.imread("input.png", cv.IMREAD_GRAYSCALE)
cv.imwrite("grayscale.png", img)

def seasoning(img):
    density_salt = 0.1
    density_pepper = 0.1

    #set salt pix
    number_of_salt = int(density_salt * (img.shape[0] * img.shape[1]))

    #add some salt
    for i in range(number_of_salt):
        y_coord = random.randint(0, img.shape[0]-1)
        x_coord = random.randint(0, img.shape[1]-1)
        img[y_coord][x_coord] = 255

    #set salt pix
    number_of_pepper = int(density_pepper * (img.shape[0] * img.shape[1]))

    #add some salt
    for i in range(number_of_pepper):
        y_coord = random.randint(0, img.shape[0]-1)
        x_coord = random.randint(0, img.shape[1]-1)
        img[y_coord][x_coord] = 0
    return img

noise_img = seasoning(img)
cv.imwrite("noise.png", noise_img)

median = cv.medianBlur(noise_img, 5)

cv.imwrite("conclution.png", median)