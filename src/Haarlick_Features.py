import cv2
import math
import mahotas
# import skimage as ski
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
    
    s_contours = sorted(contours, key = cv2.contourArea, reverse=True)

#    for i in range(10):
#        x, y, w, h = cv2.boundingRect(s_contours[i])
#        area = cv2.contourArea(s_contours[i])
#        region = thresh_img[y:y+h, x:x+w]
#        regions.append(np.asarray(region))


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

# def getHaarlickFeatures(regions):

#     features = []
#     max_reg_count=25

#     sorted_regions = sorted(regions, key=lambda x: len(x)*len(x[0]))

#     if(len(sorted_regions) < max_reg_count):
#         for region in sorted_regions:
#             textures = mahotas.features.haralick(region)
#             features.append(np.asarray(textures.mean(axis=0)))
            
#         for i in range(max_reg_count - len(regions)):
#             features.append(np.zeros(13))

#     else:
#         for i in range(max_reg_count):
#             region = regions[i]
#             textures = mahotas.features.haralick(region)
#             features.append(np.asarray(textures.mean(axis=0)))
# #    for region in regions:

# #        feature = []
    
#      #   GLCM = ski.feature.graycomatrix(region,[1,1,1,2],[math.pi/4,math.pi/2,3*math.pi/4,0],256,1)
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='contrast')))
#     #  feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='dissimilarity')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='homogeneity')))
#      #   print(GLCM[:,:,1,1])
#      #   print(ski.feature.graycoprops(GLCM, prop='energy'))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='energy')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='correlation')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='ASM')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='mean')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='variance')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='std')))
#      #   feature.append(np.mean(ski.feature.graycoprops(GLCM, prop='entropy')))
    
# #        textures = mahotas.features.haralick(region)
#    #     print(np.array(textures.mean(axis=0).shape))
# #        features.append(np.asarray(textures.mean(axis=0)))
        
#     #    features.append(np.array(feature))

# #    if(len(regions)<max_reg_count):
# #        for i in range(max_reg_count - len(regions)):
# #            features.append(np.zeros(13))

#     features=np.asarray(features)

#     return features.flatten()

def getHaarlickFeatures(regions):
    """
    Calculates Haralick features for all regions and returns the AVERAGE texture vector.
    This guarantees a fixed-length 13D feature vector per image, regardless of region count.
    """
    features = []

    for region in regions:
        # mahotas.features.haralick returns a 4x13 array (13 features in 4 directions)
        # .mean(axis=0) averages the 4 directions into a single 13-element vector
        textures = mahotas.features.haralick(region)
        features.append(textures.mean(axis=0))
        
    if not features:
        # Failsafe: If no regions > 750 were found in this image, return 13 zeros
        return np.zeros(13)

    # THE FIX: Average the features across ALL valid regions in the image.
    # The output shape is now permanently and safely exactly (13,)
    image_average_texture = np.mean(features, axis=0)

    return image_average_texture
#print(getHaarlickFeatures(regions)[1])

# def haarlickTrain(train_img_paths):
    
#     print(len(train_img_paths))
#     training_images_features=[]
#     max_len=0
#     count=0
#     for path_index in range(len(train_img_paths)):
#         count+=1
#         image = cv2.imread(train_img_paths[path_index])
#         gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         thresh_gauss = cv2.adaptiveThreshold(
#             gray_img, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             199, 5
#         )

#         thresh_gauss=cv2.bitwise_not(thresh_gauss)

#     #    print(train_img_paths[path_index])
        
#         regions,_,_ = getRegions(thresh_gauss)

#         imageFeatures=getHaarlickFeatures(regions)

#         if(len(imageFeatures)>max_len):
#             max_len=len(imageFeatures)

#         training_images_features.append(np.asarray(imageFeatures))
        
#         print(count)
    
# #    padded_train_img = []
# #    for imageFeat in training_images_features:
# #        if(len(imageFeat)<max_len):
# #            padded_train_img.append(np.pad(imageFeat, (0,(max_len-len(imageFeat))), mode='constant', constant_values=0))
# #        else:
# #            padded_train_img.append(imageFeat)

#  #   for imageFeat in training_images_features:
#  #       print(imageFeat.shape)

#  #   return padded_train_img
#     return training_images_features

def haarlickTrain(train_img_paths):
    print(f"Extracting Haralick features for {len(train_img_paths)} training images...")
    training_images_features = []
    
    for count, path in enumerate(train_img_paths, 1):
        image = cv2.imread(path)
        thresh_gauss = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 #       thresh_gauss = cv2.adaptiveThreshold(
 #           gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
 #       thresh_gauss = cv2.bitwise_not(thresh_gauss)
        
        regions, _, _ = getRegions(thresh_gauss)
        
        # This is now guaranteed to be a flat array of exactly 13 numbers
        imageFeatures = getHaarlickFeatures(regions)

        training_images_features.append(imageFeatures)
        
        if count % 500 == 0:
            print(f"Processed {count}/{len(train_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 13)
    return np.array(training_images_features)


# def haarlickTest(test_img_paths):
    
#     test_images_features=[]
#     print(len(test_img_paths))
#     count=0
#     max_len=0
#     for path_index in range(len(test_img_paths)):
#         count+=1
#         image = cv2.imread(test_img_paths[path_index])
#         gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         thresh_gauss = cv2.adaptiveThreshold(
#             gray_img, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             199, 5
#         )

#         thresh_gauss=cv2.bitwise_not(thresh_gauss)

#         regions,_,_ = getRegions(thresh_gauss)

#         imageFeatures=getHaarlickFeatures(regions)

#         if(len(imageFeatures)>max_len):
#             max_len=len(imageFeatures)

#         test_images_features.append(np.asarray(imageFeatures))
#         print(count)
    
# #    padded_test_image_feat=[]

# #    for imageFeat in test_images_features:
# #        if(len(imageFeat)<max_len):
# #            padded_test_image_feat.append(np.pad(imageFeat, (0,(max_len-len(imageFeat))), mode='constant', constant_values=0))
# #        else:
# #            padded_test_image_feat.append(imageFeat)

#   #  for imageFeat in test_images_features:
#   #      print(imageFeat.shape)

# #    return padded_test_image_feat
#     return test_images_features

def haarlickTest(test_img_paths):
    print(f"Extracting Haralick features for {len(test_img_paths)} testing images...")
    test_images_features = []
    
    for count, path in enumerate(test_img_paths, 1):
        image = cv2.imread(path)
        thresh_gauss = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#        thresh_gauss = cv2.adaptiveThreshold(
#            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
#        thresh_gauss = cv2.bitwise_not(thresh_gauss)

        regions, _, _ = getRegions(thresh_gauss)

        # This is now guaranteed to be a flat array of exactly 13 numbers
        imageFeatures = getHaarlickFeatures(regions)

        test_images_features.append(imageFeatures)
        
        # Printing every 10 instead of 500 since your test set only has 50 images
        if count % 10 == 0:
            print(f"Processed {count}/{len(test_img_paths)}")
            
    # Safely convert to a uniform 2D numpy array: shape (N_images, 13)
    return np.array(test_images_features)