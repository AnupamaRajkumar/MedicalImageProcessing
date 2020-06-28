import cv2
import numpy as np
from PIL import Image

def PreProcessing(image):
    #step 1 : resize the image to 512 x 512
    resized = cv2.resize(image, (512, 512))

    #step 2: Convert image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    #step 3: Smoothing out the noise in the image
    filImg = cv2.GaussianBlur(gray, (3,3), 0)

    #step 4: Histogram equalisation
    hist = cv2.equalizeHist(filImg)

    #step 5: Thresholding
    ret, binImg = cv2.threshold(hist, 127, 255, cv2.THRESH_BINARY)

    #cropping unnecessary height and width
    offset_x = 60
    offset_y = 70
    x = 50
    y = 10
    width = binImg.shape[0] - offset_x
    height = binImg.shape[1] - offset_y
    crop = binImg[y:height, x:width]
    #cv2.imshow("Crop", crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return crop


def Segments(crop):
    #find edges
    edged = cv2.Canny(crop, 1, 10)
    #cv2.imshow("edges", edged)

    #find contours
    cnt, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnt, key = cv2.contourArea, reverse = True)[:1]
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        hull_list.append(hull)

    #cv2.drawContours(edged, cnts, -1, (0,0,255), 2)
    for i in range(len(cnts)):
        cv2.drawContours(edged, hull_list, -1, (0,0,255), 2)

    cv2.imshow("Hull", edged)

    #opening operation
    kernel = np.zeros((7,7),np.uint8)
    openImg = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Open", openImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



path =  'D:\EIT_AUS_TUB\SoSe2020_MLInMIP\Images\Image_7.jpeg'
#step1 : Read image
image = cv2.imread(path)
cv2.imshow("Image", image)
#preprocessing image
crop = PreProcessing(image)

#lung segments
Segments(crop)

cv2.waitKey(0)
cv2.destroyAllWindows()








