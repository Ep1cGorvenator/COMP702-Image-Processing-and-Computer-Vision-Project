import cv2
import math
import mahotas
import skimage as ski
import numpy as np
import sys


#image = cv2.imread('dataset/raw/R10/new/front/R10_new_front_001.jpg')
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresh_gauss = cv2.adaptiveThreshold(
#    gray_image, 255,
#    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#    cv2.THRESH_BINARY,
#    199, 5
#)

#thresh_gauss=cv2.bitwise_not(thresh_gauss)

def getRegions(thresh_img):
    
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    dims = []
    indices = []
    count=0
#    if(len(contours)==1):
#        cv2.imshow("window", thresh_img)
#        cv2.waitKey(0)
#        sys.exit()
    
 #   s_contours = sorted(contours, key = cv2.contourArea, reverse=True)

    for contour in contours:
        count+=1
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
    
        if area > 750:
            dims.append([w,h])
            indices.append(count)
            region = thresh_img[y:y+h, x:x+w]
            regions.append(np.asarray(region))
            count+=1

    return regions,dims,indices


#regions,dims,indices = getRegions(thresh_gauss)

#print(np.max(regions[17]))
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
    
     #   GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='contrast')))
    #  feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='dissimilarity')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='homogeneity')))
     #   print(GLCM[:,:,1,1])
     #   print(ski.feature.graycoprops(GLCM, prop='energy'))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='energy')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='correlation')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='ASM')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='mean')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='variance')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='std')))
     #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='entropy')))
    
        textures = mahotas.features.haralick(region)
   #     print(np.array(textures.mean(axis=0).shape))
        features.append(np.asarray(textures.mean(axis=0)))
        
    #    features.append(np.array(feature))
    features=np.asarray(features)

    return features.flatten()

#print(getHaarlickFeatures(regions)[1])

def haarlickTrain(train_img_paths):
    
    print(len(train_img_paths))
    training_images_features=[]
    max_len=0
    count=0
    for path_index in range(len(train_img_paths)):
        count+=1
        image = cv2.imread(train_img_paths[path_index])
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_gauss = cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            199, 5
        )

        thresh_gauss=cv2.bitwise_not(thresh_gauss)

    #    print(train_img_paths[path_index])
        
        regions,_,_ = getRegions(thresh_gauss)

        imageFeatures=getHaarlickFeatures(regions)

        if(len(imageFeatures)>max_len):
            max_len=len(imageFeatures)

        training_images_features.append(np.asarray(imageFeatures))
        
        print(count)
    
    for imageFeat in training_images_features:
        if(len(imageFeat)<max_len):
            np.pad(imageFeat, (0,len(max_len-imageFeat)), mode='constant', constant_values=0)

    return training_images_features


def haarlickTest(test_img_paths):
    
    test_images_features=[]
    print(len(test_img_paths))
    count=0
    max_len=0
    for path_index in range(len(test_img_paths)):
        count+=1
        image = cv2.imread(test_img_paths[path_index])
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_gauss = cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            199, 5
        )

        thresh_gauss=cv2.bitwise_not(thresh_gauss)

        regions,_,_ = getRegions(thresh_gauss)

        imageFeatures=getHaarlickFeatures(regions)

        if(len(imageFeatures)>max_len):
            max_len=len(imageFeatures)

        test_images_features.append(np.asarray(imageFeatures))
        print(count)
    

    for imageFeat in test_images_features:
        if(len(imageFeat)<max_len):
            np.pad(imageFeat, (0,len(max_len-imageFeat)), mode='constant', constant_values=0)
            
    return test_images_features