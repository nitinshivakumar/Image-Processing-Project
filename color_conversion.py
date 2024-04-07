#Import Packages

import numpy as np
import cv2
import copy
import math

"""
2.1 Part A: Color Conversion (40 Points)

Given a RGB image (Lenna.png provided in homework 3.zip), your program should be able to conduct the following operations and output a resultant image.

Convert RGB to HSV: Convert the original image (Lenna.png) to HSV color space and save the output as hsv image 1.png (10 points) (use formula mentioned in link).

"""
image = cv2.imread('Lenna.png')
 
image = image.astype(np.float32)/ 255.0
 
B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
 
V= np.max(image, axis = 2)
S = np.zeros_like(V)
H = np.zeros_like(V)
min_matrix = np.min(image, axis = 2)

not_zero = (V != 0)
S[not_zero] = (V[not_zero] - min_matrix[not_zero]) / V[not_zero]

H[V == R] = (60.0 * (G[V == R] - B[V == R])) / (V[V == R] - min_matrix[V == R])
H[V == G] = (120.0 + 60.0 * (B[V == G] - R[V == G])) / (V[V == G] - min_matrix[V == G])
H[V == B] = (240.0 + 60.0 * (R[V == B] - G[V == B])) / (V[V == B] - min_matrix[V == B])
H[(V == R) & (V == G) & (V == B)] = 0

H[H<0] = H[H<0] + 360

H = (H / 2)
S = (S * 255)
V = (V * 255)
 
hsv_image = np.dstack([H, S, V])
cv2.imwrite('hsv_image_1.png', hsv_image)

"""

Convert RGB to HSV: Convert the original image (Lenna.png) to HSV color space and save the output as hsv image 2.png (10 points) (use formula mentioned in class).

"""
image = cv2.imread('Lenna.png')

image = image.astype(np.float32)/ 255.0

B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]

S = np.zeros_like(B)
V = np.zeros_like(B)
H = np.zeros_like(B)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        V[i, j] = (R[i, j] + G[i, j] + B[i, j])/3
        S[i, j] = 1 - ((3/(R[i, j] + G[i, j] + B[i, j])) * ((min(R[i, j], G[i, j], B[i, j]))))
        up = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
        down = np.sqrt((R[i, j] - G[i, j])**2 + ((R[i, j] - B[i, j])*(G[i, j] - B[i, j])))
        theta = np.arccos((up/down))
        if  B[i, j] <= G[i, j]:
            H[i, j] = theta
        else:
            H[i, j] = ((360) - theta)

H = H/2
S = S * 255
V = V * 255
hsv_image = np.dstack([H, S, V])
 
cv2.imwrite('hsv_image_2.png', hsv_image)

"""

Convert RGB to CMYK: Convert the original image (Lenna.png) to CMYK color space and save the output as cmyk image.png (10 points). (use formula mentioned in class); (Use R=C, G=M, B=Y, A=K and store RGBA image as png.)

"""

rgb_image = cv2.imread('Lenna.png')  
rgb_image = rgb_image.astype(np.float32) / 255.0
R, G, B = rgb_image[:, :, 2], rgb_image[:, :, 1], rgb_image[:, :, 0]
Y = 1 - B
M = 1 - G
C = 1 - R
K = np.minimum(C, np.minimum(M, Y))
C[K == 1] = 0
M[K == 1] = 0
Y[K == 1] = 0
C = (C - K) / (1 - K)
M = (M - K) / (1 - K)
Y = (Y - K) / (1 - K)
cmyk_image = np.dstack([C, M, Y, K])
cv2.imwrite('cmyk_image.png', (cmyk_image * 255))

"""

Convert RGB to LAB: Convert the original image (Lenna.png) to LAB color space and save the output as lab image.png (use formula mentioned in link) (10 points).

"""

image = cv2.imread('Lenna.png')
 
image = image.astype(np.float32)

B, G, R = image[:, :, 0]/255, image[:, :, 1]/255, image[:, :, 2]/255

def f(t):
    if t > 0.008856:
        n = t**(1/3)
    else:
        n = (7.787 * t) + 16/116
    return n

vector = np.zeros_like(image)
Lab = np.zeros_like(image)

vector[:, :, 0] = (0.412453 * R) + (0.357580 * G) + (0.180423 * B)
vector[:, :, 1] = (0.212671 * R) + (0.715160 * G) + (0.072169 * B)
vector[:, :, 2] = (0.019334 * R) + (0.119193 * G) + (0.950227 * B)
    
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        vector[:,:,0][i][j] = vector[:,:,0][i][j]/0.950456
        vector[:,:,2][i][j] = vector[:,:,2][i][j]/1.088754
        
        if vector[:,:,1][i][j]>0.008856:
            Lab[:,:,0][i][j] = (116 * (vector[:,:,1][i][j]**(1/3))) - 16
        elif vector[:,:,1][i][j]<=0.008856:
            Lab[:,:,0][i][j] = vector[:,:,1][i][j] * 903.3
        Lab[:,:,1][i][j] = (500 *(f(vector[:,:,0][i][j]) - f(vector[:,:,1][i][j]))) 
        Lab[:,:,2][i][j] = (200 *(f(vector[:,:,1][i][j]) - f(vector[:,:,2][i][j]))) 

Lab[:,:,0] = ((Lab[:,:,0]  * 255)/100).astype(np.uint8)
Lab[:,:,1] = ((Lab[:,:,1] )+128).astype(np.uint8)
Lab[:,:,2] = ((Lab[:,:,2] )+128).astype(np.uint8)

cv2.imwrite('lab_image.png', Lab)