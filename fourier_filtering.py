#Import Packages

import numpy as np  
import cv2 as cv
import copy
import math


"""

2.3 Part C: Gaussian Filter Smoothing using Fourier transform (20 Points)
Given a gray-scale image (Noisy image.png provided in homework 3.zip), your program should be able to do the following steps.
Convert gray-scale image to Fourier domain: Convert the original image (Noisy image.png) to Fourier domain and save the output as converted fourier.png (10 points).

"""

img = cv.imread('Noisy_image.png', cv.IMREAD_GRAYSCALE)
width = img.shape[0]
height = img.shape[1]
pad = np.zeros((width*2, height*2), dtype=np.float32)
new_filter = np.zeros_like(pad)
fourier_to_center = copy.deepcopy(pad)

pad[:img.shape[0], :img.shape[1]] = img

x, y = np.indices(new_filter.shape)
new_filter = pad * (-1) ** (x + y)
dft1 = np.fft.fft2(new_filter)
dft = cv.dft(np.float32(new_filter), flags=cv.DFT_COMPLEX_OUTPUT)
spect = np.log(cv.magnitude(dft[:, :, 0], dft[:, :, 1]) + 1)
spect = cv.normalize(spect, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

cv.imwrite('converted_fourier.png', spect) 

"""

• Gaussian Filter smoothing: First, apply a low-pass filter (you can choose the filter that achieves reasonable results) in the Fourier domain. Then, invert the image to gray-scale space and save the output as gaussian fourier.png (10 points).
Note: The results of guassian fourier.png and gaussian image.png (from part B) should look very similar in terms of outputs.

"""

low_pass_filter = copy.deepcopy(pad)
x, y = np.indices(low_pass_filter.shape)
center_x = pad.shape[0] / 2
center_y = pad.shape[1] / 2
distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
low_pass_filter = np.where(distance <= 112, 1, 0)
product = dft1  * low_pass_filter 
shift_product = np.fft.ifftshift(product)
pad_original = np.abs(np.fft.ifft2(shift_product))
original = pad_original[:width, :height]
image_without_pad = original[:img.shape[0], :img.shape[1]]
cv.imwrite('guassian_fourier.png', image_without_pad) 

