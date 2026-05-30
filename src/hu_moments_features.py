import cv2
import math
import mahotas
# import skimage as ski
import numpy as np
import sys


def getRegions(thresh_img):
    
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    dims = []
    indices = []
    count=0
    
    s_contours = sorted(contours, key = cv2.contourArea, reverse=True)

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

    return regions, dims, indices


def getHuMomentFeatures(regions):

    features = []

    for region in regions:
        moments = cv2.moments(region)
        huMoments = cv2.HuMoments(moments).flatten()
        features.append(huMoments)
        
    if not features:
        # Failsafe: If no regions > 750 were found in this image, return 7 zeros
        return np.zeros(7)

    # THE FIX: Average the features across ALL valid regions in the image.
    # The output shape is now permanently and safely exactly (7,)
    image_average_texture = np.mean(features, axis=0)

    return image_average_texture


def huMomentTrainAdaptThresh(train_img_paths):
    print(f"Extracting Hu Moment features for {len(train_img_paths)} training images...")
    training_images_features = []
    
    for count, path in enumerate(train_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_gauss = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        thresh_gauss = cv2.bitwise_not(thresh_gauss)
        
        regions, _, _ = getRegions(thresh_gauss)
        
        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        training_images_features.append(imageFeatures)
        
        if count % 500 == 0:
            print(f"Processed {count}/{len(train_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7 )
    return np.array(training_images_features)


def huMomentTestAdaptThresh(test_img_paths):
    print(f"Extracting Hu moment features for {len(test_img_paths)} testing images...")
    test_images_features = []
    
    for count, path in enumerate(test_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_gauss = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        thresh_gauss = cv2.bitwise_not(thresh_gauss)

        regions, _, _ = getRegions(thresh_gauss)

        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        test_images_features.append(imageFeatures)
        
        # Printing every 10 instead of 500 since your test set only has 50 images
        if count % 10 == 0:
            print(f"Processed {count}/{len(test_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7)
    return np.array(test_images_features)




def huMomentTrainOtsuThresh(train_img_paths):
    print(f"Extracting Hu Moment features for {len(train_img_paths)} training images...")
    training_images_features = []
    
    for count, path in enumerate(train_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, otsu_thresh = cv2.threshold( gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        regions, _, _ = getRegions(otsu_thresh)
        
        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        training_images_features.append(imageFeatures)
        
        if count % 500 == 0:
            print(f"Processed {count}/{len(train_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7 )
    return np.array(training_images_features)


def huMomentTestOtsuThresh(test_img_paths):
    print(f"Extracting Hu moment features for {len(test_img_paths)} testing images...")
    test_images_features = []
    
    for count, path in enumerate(test_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, otsu_thresh = cv2.threshold( gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        regions, _, _ = getRegions(otsu_thresh)

        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        test_images_features.append(imageFeatures)
        
        # Printing every 10 instead of 500 since your test set only has 50 images
        if count % 10 == 0:
            print(f"Processed {count}/{len(test_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7)
    return np.array(test_images_features)

def huMomentTrain(train_img_paths):
    print(f"Extracting Hu Moment features for {len(train_img_paths)} training images...")
    training_images_features = []
    
    for count, path in enumerate(train_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        regions, _, _ = getRegions(gray_img)
        
        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        training_images_features.append(imageFeatures)
        
        if count % 500 == 0:
            print(f"Processed {count}/{len(train_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7 )
    return np.array(training_images_features)


def huMomentTest(test_img_paths):
    print(f"Extracting Hu moment features for {len(test_img_paths)} testing images...")
    test_images_features = []
    
    for count, path in enumerate(test_img_paths, 1):
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        regions, _, _ = getRegions(gray_img)

        # This is now guaranteed to be a flat array of exactly 7 numbers
        imageFeatures = getHuMomentFeatures(regions)

        test_images_features.append(imageFeatures)
        
        # Printing every 10 instead of 500 since your test set only has 50 images
        if count % 10 == 0:
            print(f"Processed {count}/{len(test_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 7)
    return np.array(test_images_features)