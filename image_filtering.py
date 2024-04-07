#Import Packages

import numpy as np
import cv2
import copy
import math

"""

2.2 Part B: Image Filtering (40 Points)
Given a gray-scale image (Noisy image.png provided in homework 3.zip), your program should be able to apply the filter mentioned and output a resultant image.
• Convolution: Convolve the following filter to the original image (Noisy image.png) and save the output as convolved image.png (7.5 points).

"""

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

def convulate(image, i, j):
    out = np.array([[image[i+1][j+1], image[i+1][j], image[i+1][j-1]],
                    [image[i][j+1], image[i-1][j], image[i][j-1]],
                    [image[i-1][j+1], image[i-1][j], image[i-1][j-1]]]) * (np.array([[1, 1, 1],
                                                                                    [1, 1, 1],
                                                                                    [1, 1, 1]])/ 9)
    
    out = np.sum(out)
    return out

conv2d = np.zeros([image.shape[0] + 2, image.shape[1] + 2])
conv2d[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1] = image
new = np.zeros_like(conv2d)

for i in range(1, conv2d.shape[0]-1):
    for j in range(1, conv2d.shape[1]-1):
        new[i, j] = convulate(conv2d, i, j)

unpad_conv2d = new[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1]
cv2.imwrite('convolved_image.png', unpad_conv2d)               

"""

Averaging Filter: Apply the following filter to the original image (Noisy image.png) and save the output as average image.png (7.5 points).

"""

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

def corelate_filter(image, i, j):
    out = np.array([[image[i-1][j-1], image[i-1][j], image[i-1][j+1]],
                    [image[i][j-1], image[i-1][j], image[i][j+1]],
                    [image[i+1][j-1], image[i+1][j], image[i+1][j+1]]]) * (np.array([[1, 1, 1],
                                                                                    [1, 1, 1],
                                                                                    [1, 1, 1]])/ 9)
    
    out = np.sum(out)
    return out

conv2d = np.zeros([image.shape[0] + 2, image.shape[1] + 2])
conv2d[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1] = image
new = np.zeros_like(conv2d)

for i in range(1, conv2d.shape[0]-1):
    for j in range(1, conv2d.shape[1]-1):
        new[i, j] = corelate_filter(conv2d, i, j)

unpad_conv2d = new[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1]
cv2.imwrite('average_image.png', unpad_conv2d)         

'''

Gaussian Filter: Apply the following filter to the original image (Noisy image.png) and save the output as gaussian image.png (7.5 points).

'''

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

def gaussian_filter(image, i, j):
    out = np.array([[image[i-1][j-1], image[i-1][j], image[i-1][j+1]],
                    [image[i][j-1], image[i-1][j], image[i][j+1]],
                    [image[i+1][j+1], image[i+1][j], image[i+1][j+1]]]) * (np.array([[1, 2, 1],
                                                                                    [2, 4, 2],
                                                                                    [1, 2, 1]])/ 16)
    
    out = np.sum(out)
    return out

conv2d = np.zeros([image.shape[0] + 2, image.shape[1] + 2])
conv2d[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1] = image
new = np.zeros_like(conv2d)

for i in range(1, conv2d.shape[0]-1):
    for j in range(1, conv2d.shape[1]-1):
        new[i, j] = gaussian_filter(conv2d, i, j)

unpad_conv2d = new[1:conv2d.shape[0]-1, 1:conv2d.shape[1]-1]
cv2.imwrite('gaussian_image.png', unpad_conv2d)         

"""

Median Filter: Apply a 5×5 median filter to the original image (Noisy image.png) and save the output as median image.png (7.5 points)

"""

image = cv2.imread('Noisy_image.png', cv2.IMREAD_GRAYSCALE)

def median_filter(image, i, j):
    out = np.array([[image[i-2][j-2], image[i-2][j-1], image[i-2][j], image[i-2][j+1], image[i-2][j+2]],
                    [image[i-1][j-2], image[i-1][j-1], image[i-1][j], image[i-2][j+1], image[i-2][j+2]],
                    [image[i][j-2], image[i][j-1], image[i][j], image[i][j+1], image[i][j+2]],
                    [image[i+1][j-2], image[i+1][j-1], image[i+1][j], image[i+1][j+1], image[i+1][j+2]],
                    [image[i+2][j-2], image[i+2][j-1], image[i+2][j], image[i+2][j+1], image[i+2][j+2]]])
    
    out = np.sort(out.ravel())
    return out[12]

conv2d = np.zeros([image.shape[0] + 4, image.shape[1] + 4])
conv2d[2:conv2d.shape[0]-2, 2:conv2d.shape[1]-2] = image
new = np.zeros_like(conv2d)

for i in range(2, conv2d.shape[0]-2):
    for j in range(2, conv2d.shape[1]-2):
        new[i, j] = median_filter(conv2d, i, j)

unpad_conv2d = new[2:conv2d.shape[0]-2, 2:conv2d.shape[1]-2]
cv2.imwrite('median_image.png', unpad_conv2d)         

"""

Contrast and Brightness Adjustment: Apply contrast and brightness adjustment so that the original image (Uexposed.png in homework 3.zip) changes to high contrast and brighter image (changes should be visibly perceivable) and save the output as adjusted image.png (10 points).

"""

contrast = 2.3
brightness = 80
image = cv2.imread('Uexposed.png')
for b in range(image.shape[0]):
    for g in range(image.shape[1]):
        for r in range(image.shape[2]):
            image[b,g,r] = np.clip(contrast*image[b,g,r] + brightness, 0, 255)
cv2.imwrite('adjusted_image.png', image)     