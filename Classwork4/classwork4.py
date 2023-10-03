import numpy as np
import cv2 as cv

img = cv.imread("input.png", cv.IMREAD_GRAYSCALE)

# ทำ Fourier Transform 2 มิติ
fft_img = np.fft.fft2(img)

# เชื่อมโยงข้อมูลความถี่ให้อยู่ในรูปแบบสเปกตรัม (Spectrum)
fft_shifted = np.fft.fftshift(fft_img)

# คำนวณ Magnitude Spectrum (Amplitude Spectrum)
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

# Horizontal Sobel Filter
horizontal_sobel_filter = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

horizontal_gradient = cv.filter2D(img, -1, horizontal_sobel_filter)
gradient_magnitude = np.sqrt(horizontal_gradient**2)

# Optional: Normalize the magnitude to enhance visibility
gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
gradient_magnitude = (gradient_magnitude * 255).astype(np.uint8)

cv.imwrite("gray.png", img)
cv.imwrite("Sobel.png", gradient_magnitude)
cv.imwrite("FDomain.png", magnitude_spectrum.astype(np.uint8))


