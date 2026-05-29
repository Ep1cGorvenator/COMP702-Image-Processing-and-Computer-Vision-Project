import cv2
import math
import skimage as ski
import numpy as np

image = cv2.imread('dataset/raw/R10/new/front/R10_new_front_001.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh_gauss = cv2.adaptiveThreshold(
    gray_image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    199, 5
)

thresh_gauss=cv2.bitwise_not(thresh_gauss)

def getRegions(thresh_img):
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    dims = []
    indices = []
    count=0
    for contour in contours:
        count+=1
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
    
        if area > 500:
            dims.append([w,h])
            indices.append(count)
            region = thresh_img[y:y+h, x:x+w]
            regions.append(np.array(region))
            count+=1

    return regions,dims,indices


regions,dims,indices = getRegions(thresh_gauss)

print(np.max(regions[17]))
#print(max(dims))
#print(len(indices))

#print("Number of regions: ",len(regions))

#cv2.imshow("window", regions[5])
##cv2.imshow("window", thresh_gauss)
#cv2.waitKey(0)

#GLCM = ski.feature.graycomatrix(regions[17],[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
#result = ski.feature.graycoprops(GLCM, 'contrast')
#np.average(result)

def getHaarlickFeatures(regions):

    features = []

    for region in regions:

        feature = []
    
        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='contrast')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='dissimilarity')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='homogeneity')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
     #   print(GLCM[:,:,1,1])
     #   print(ski.feature.graycoprops(GLCM, prop='energy'))
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='energy')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='correlation')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='ASM')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='mean')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='variance')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='std')))

        GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
        feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='entropy')))

        features.append(np.array(feature))

    return features

print(getHaarlickFeatures(regions)[1])

