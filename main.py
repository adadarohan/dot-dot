import cv2
from tqdm import tqdm
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

###### STAGE ONE ######
# Pre proccessing + edge detection

# Read the original image
imgunr = cv2.imread('image2.jpg') 
r = 200.0 / imgunr.shape[0]
dim = (int(imgunr.shape[1] * r), 200)
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

###### STAGE TWO ######
# Coverting edges into dots

dots = []

shape = [len(edges), len(edges[0])]

for x in range(shape[0]) :
    for y in range(shape[1]) :
        if edges[x][y] == 255 :
            dots.append([y,x])

###### STAGE THREE  ######
# find path connecting dots
print('finding shortest path')

sorted_dots = []
cdot = []

ns = [0,0]

def pyth_dist(elem) :
    global ns
    return math.sqrt(((elem[0] - ns[0] )**2) + ((elem[1] - ns[1])**2))

for i in tqdm(range(len(dots))):
    nearl = sorted(dots, key=pyth_dist)
    near = []
    for n in nearl : 
        if n not in sorted_dots :
            near = n
            break
    ns = near
    sorted_dots.append(near)

blank_image = np.zeros((shape[0],shape[1], 3), np.uint8)

pts = np.array(sorted_dots, np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(blank_image,[pts],True,(0,255,255))


# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection with Line Now', blank_image)
cv2.waitKey(0)

print(len(sorted_dots))

###### STAGE FOUR ######
# remove dots which are in a line 

fdots = sorted_dots.copy()

print('eliminating straight dots')
for i in tqdm(range(1, len(sorted_dots)-2)):
    try :
        pgrad = (sorted_dots[i+1][1] - sorted_dots[i][1]) / (sorted_dots[i+1][0] - sorted_dots[i][0])
    except:
        pgrad = 1000000
    
    try :
        fgrad = (sorted_dots[i][1] - sorted_dots[i-1][1]) / (sorted_dots[i][0] - sorted_dots[i-1][0])
    except:
        fgrad = 1000000

    dist = math.sqrt(((sorted_dots[i-1][0] - sorted_dots[i][0] )**2) + ((sorted_dots[i-1][1] - sorted_dots[i][1])**2))

    if abs(pgrad - fgrad) < 15 or dist < 1: #if the change in gradient is less than 0.2
        fdots.remove(sorted_dots[i]) #remove the middle one



blank_image = np.zeros((shape[0],shape[1], 3), np.uint8)

pts = np.array(fdots, np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(blank_image,[pts],True,(0,255,255))


# Display Canny Edge Detection Image
cv2.imshow('Removed Straights', blank_image)
cv2.waitKey(0)

print(len(fdots))

cv2.destroyAllWindows()