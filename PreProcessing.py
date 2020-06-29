import cv2
import numpy as np
from PIL import Image

def PreProcessing(image):
  
    #step 2: Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #step 3: Smoothing out the noise in the image
    filImg = cv2.GaussianBlur(gray, (5,5), 6)

    #step 4: Histogram equalisation
    hist = cv2.equalizeHist(filImg)

    #step 5: Thresholding
    ret, binImg = cv2.threshold(hist, 120, 255, cv2.THRESH_BINARY)

    #cv2.imshow("hist", binImg)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return hist, binImg


def Segments(crop):

    #opening operation
    edged = cv2.Canny(crop, 0, 255)
    print(len(edged))
    #cv2.imshow("binary", edged)
  
    #find contours
    cnt, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cnt))
    kernel = np.zeros((10,10),np.uint8)
    #openImg = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    openImg = cv2.dilate(edged, kernel)
    openImg = cv2.dilate(openImg, kernel)
    print(len(cnt))
    cnt, hierarchy = cv2.findContours(openImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnt, key = cv2.contourArea, reverse = True)[:len(cnt)]

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(cnts)):       
        #print(cv2.contourArea(cnts[i]))
        hull = cv2.convexHull(cnts[i])
        hull_list.append(hull)

    #print(len(hull_list))
    cv2.drawContours(openImg, hull_list, -1, (255,255,255), -1)

    cv2.imshow("Hull", openImg)
    return openImg


path =  'D:\EIT_AUS_TUB\SoSe2020_MLInMIP\Images\Image_1.jpeg'
#step1 : Read image
image = cv2.imread(path)
#cv2.imshow("Image", image)
#preprocessing image
resized = cv2.resize(image, (512, 512))
hist, binImg = PreProcessing(resized)

#lung segments
segments = Segments(binImg)

overlap = cv2.bitwise_and(hist, segments)

cv2.imshow("Hist", hist)

cv2.imshow("Overlap", overlap)

cv2.waitKey(0)
cv2.destroyAllWindows()








