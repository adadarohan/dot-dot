import cv2
from tqdm import tqdm
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

### Streamlit stuff
st.set_page_config(page_title='Dot to Dot Maker', page_icon='✏️', layout="centered", initial_sidebar_state="auto", menu_items=None)


st.title('Dot to Dot Maker')
st.write('Hi there! To begin, upload an image which you want to convert to a dot to dot.')

st.header('Convert your image!')
container = st.container()
uploaded_file = container.file_uploader("Upload an image file")
my_bar = container.progress(0)

stats = st.container()
stats.header('Stats for nerds 😎')
col0, col1, col2, col3 = stats.columns(4)

st.sidebar.markdown('This website was built by [Rohan Kapur](https://github.com/adadarohan) and inspired by [Thomas Pavitte](https://thomasmakesstuff.com/)\'s 1000 dot-to-dot book')


###### STAGE ONE ######
# Pre proccessing + edge detection
# Read the original image


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgunr = cv2.imdecode(file_bytes, 1)
    #imgunr = cv2.imread(uploaded_file) 
    with col0:
        st.image(imgunr, caption="Original Image", use_column_width='always', channels="BGR")

    r = 200.0 / imgunr.shape[0]
    dim = (int(imgunr.shape[1] * r), 200)
    img = cv2.resize(imgunr, dim, interpolation=cv2.INTER_AREA)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    # # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)
    with col1:
        st.image(edges, caption="Traced Edges", use_column_width='always')

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
        my_bar.progress(i/len(dots))

    blank_image = np.zeros((shape[0],shape[1], 3), np.uint8)

    pts = np.array(sorted_dots, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(blank_image,[pts],True,(0,255,255))


    # # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection with Line Now', blank_image)
    # cv2.waitKey(0)

    print(len(sorted_dots))

    ###### STAGE FOUR ######
    # remove dots which are in a line 

    fdots = sorted_dots.copy()

    print('eliminating straight dots')
    for i in tqdm(range(1, len(sorted_dots)-2)):
        last_dot = sorted_dots[i-1]
        if i != 1 : 
            for z in reversed(range(0, i-1)) :
                if sorted_dots[z] in fdots :
                    last_dot = sorted_dots[z]
                    break

        try :
            pgrad = (sorted_dots[i+1][1] - sorted_dots[i][1]) / (sorted_dots[i+1][0] - sorted_dots[i][0])
        except:
            pgrad = 1000000
        
        try :
            fgrad = (sorted_dots[i][1] - last_dot[1]) / (sorted_dots[i][0] - last_dot[0])
        except:
            fgrad = 1000000

        dist = math.sqrt(((last_dot[0] - sorted_dots[i][0] )**2) + ((last_dot[1] - sorted_dots[i][1])**2))

        if abs(pgrad - fgrad) < 10 or dist < 1.1 : #if the change in gradient is less than 0.2
            fdots.remove(sorted_dots[i]) #remove the middle one


    # Resize image to make it 3x
    sf = 10
    blank_image = np.zeros((shape[0]*sf,shape[1]*sf, 3), np.uint8)
    blank_image.fill(255)
    rfdots = [[z[0]*sf, z[1]*sf] for z in fdots]
    pts = np.array(rfdots, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(blank_image,[pts],True,(255,0,0))

    # # Display Canny Edge Detection Image
    # cv2.imshow('Removed Straights', blank_image)
    # cv2.waitKey(0)
    with col2:
        st.image(blank_image, caption="Output Lines",  use_column_width='always', channels="BGR")

    cv2.imwrite('C:\code\dot-dot\output\lines.jpg',blank_image)

    print(f'Total Number of dots in image : {len(rfdots)}')
    stats.write(f'Number of dots in inital edges : {len(dots)}')
    stats.write(f'Number of dots pre-proccessing : {len(dots)}')
    stats.write(f'Number of dots in output image : {len(rfdots)}')

    ###### STAGE FIVE ######
    # format and output
    blank_image = np.zeros((shape[0]*sf,shape[1]*sf, 3), np.uint8)
    blank_image.fill(255)
    colors = [{"name":"Flickr Pink","hex":"f72585","rgb":[247,37,133],"cmyk":[0,85,46,3],"hsb":[333,85,97],"hsl":[333,93,56],"lab":[55,79,1]},{"name":"Byzantine","hex":"b5179e","rgb":[181,23,158],"cmyk":[0,87,13,29],"hsb":[309,87,71],"hsl":[309,77,40],"lab":[43,70,-34]},{"name":"Purple","hex":"7209b7","rgb":[114,9,183],"cmyk":[38,95,0,28],"hsb":[276,95,72],"hsl":[276,91,38],"lab":[32,66,-66]},{"name":"Purple","hex":"560bad","rgb":[86,11,173],"cmyk":[50,94,0,32],"hsb":[268,94,68],"hsl":[268,88,36],"lab":[27,60,-68]},{"name":"Trypan Blue","hex":"480ca8","rgb":[72,12,168],"cmyk":[57,93,0,34],"hsb":[263,93,66],"hsl":[263,87,35],"lab":[25,58,-69]},{"name":"Trypan Blue","hex":"3a0ca3","rgb":[58,12,163],"cmyk":[64,93,0,36],"hsb":[258,93,64],"hsl":[258,86,34],"lab":[23,55,-70]},{"name":"Persian Blue","hex":"3f37c9","rgb":[63,55,201],"cmyk":[69,73,0,21],"hsb":[243,73,79],"hsl":[243,57,50],"lab":[34,48,-74]},{"name":"Ultramarine Blue","hex":"4361ee","rgb":[67,97,238],"cmyk":[72,59,0,7],"hsb":[229,72,93],"hsl":[229,83,60],"lab":[47,36,-74]},{"name":"Dodger Blue","hex":"4895ef","rgb":[72,149,239],"cmyk":[70,38,0,6],"hsb":[212,70,94],"hsl":[212,84,61],"lab":[61,5,-52]},{"name":"Vivid Sky Blue","hex":"4cc9f0","rgb":[76,201,240],"cmyk":[68,16,0,6],"hsb":[194,68,94],"hsl":[194,85,62],"lab":[76,-22,-29]}]
    
    print('formatting')
    for i in tqdm(range(len(rfdots))):
        if i < 100 * len(colors) :
            color_ar = colors[i//100]['rgb']
            c = (color_ar[2], color_ar[1] , color_ar[0],)
        else :
            c = (201,240,76)
        blank_image = cv2.circle(blank_image, (rfdots[i][0],rfdots[i][1]), radius=1, color=c, thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blank_image, str(i) ,(rfdots[i][0] + 5, rfdots[i][1] + 5), font, 0.4 ,c ,0, cv2.LINE_AA)

    # # Display Canny Edge Detection Image
    # cv2.imshow('Added Dots', blank_image)
    # cv2.waitKey(0)
    with col3:
        st.image(blank_image, caption="Output Dots & Numbers",  use_column_width='always', channels="BGR")

    my_bar.progress(1.0)
    cv2.imwrite('dots.jpg',blank_image)

    with open("dots.jpg", "rb") as file:
        btn = container.download_button(
            label="Download dot-to-dot",
            data=file,
            file_name="dots.jpg",
            mime="image/jpg"
        )
    #cv2.destroyAllWindows()

