import cv2
from tqdm import tqdm
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

###### STAGE ONE ######
# Pre proccessing + edge detection

# Read the original image
imgunr = cv2.imread('images\image2.jpg') 
r = 500.0 / imgunr.shape[0]
dim = (int(imgunr.shape[1] * r), 500)
img = cv2.resize(imgunr, dim, interpolation=cv2.INTER_AREA)

# # Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)


# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

img_copy = img.copy()
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
shape = [len(edges), len(edges[0])]

blank_image = np.zeros((shape[0],shape[1], 3), np.uint8)

cv2.drawContours(blank_image, contours, -1, (255,255,255), 1)

# # Display original image
cv2.imshow('Contours', blank_image)
cv2.waitKey(0)

print(len(edges), len(contours)*2)

uc = [list(z[0]) for x in contours for z in x ]
res = []
[res.append(x) for x in uc if x not in res]
print(len(res))
cv2.destroyAllWindows()