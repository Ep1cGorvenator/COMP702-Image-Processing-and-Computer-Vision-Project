# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathlib import Path

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

def extract_hu_moments(image):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the Hu Moments feature vector
	moments = cv2.moments(gray)
	huMoments = cv2.HuMoments(moments).flatten()
	# return the Hu Moments as the feature vector
	return huMoments

# grab the list of images that we'll be describing
print("[INFO] describing images...")
BASE_DIR = Path(__file__).resolve().parent.parent
imagePaths = list(paths.list_images(BASE_DIR / "dataset" / "processed"))

# initialize the raw pixel intensities matrix, the histogram features matrix, and HU Moments features matrix
rawImages = []
features = []
labels = []
HUMoments = []

#----------------FEATURE EXTRACTION----------------
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	image_file = imagePath.rsplit('\\', 1)[-1].rsplit('.jpeg', 1)[0]
	
	if "R100" in image_file:
		label = "R100"
	elif "R200" in image_file:	
		label = "R200"
	elif "R50" in image_file:
		label = "R50"
	elif "R10" in image_file:
		label = "R10"	
	elif "R20" in image_file:
		label = "R20"
	# extract raw pixel intensity "features"
	pixels = image_to_feature_vector(image)
	# extract color histogram to characterize the color distribution of the pixels in the image
	hist = extract_color_histogram(image)

	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	HUMoments.append(extract_hu_moments(image))

	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
		
# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
HUMoments = np.array(HUMoments)
labels = np.array(labels)

print("\nHow much memory is being consumed by the raw images matrix and features matrix:")
print("pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))
print("Hu Moments matrix: {:.2f}MB".format(
	HUMoments.nbytes / (1024 * 1000.0)))

#---------TRAIN TEST SPLIT----------------
#Using raw pixel intensities as features for training and testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.20, random_state=42)

#Using color histograms as features for training and testing
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.20, random_state=42)

#Using HU Moments as features for training and testing
(trainHM, testHM, trainLabelsHM, testLabelsHM) = train_test_split(
	HUMoments, labels, test_size=0.20, random_state=42)

#----------------KNN CLASSIFICATION----------------
print("\n-------------------KNN CLASSIFICATION-------------------")
#YOU CAN SPECIFY HOW MANY NEIGHBOURS TO USE WITH THE n_neighbors PARAMETER, 
# AND HOW MANY CPU CORES TO USE WITH THE n_jobs PARAMETER

#RAW PIXEL FEATURES
print("\nevaluating raw pixel accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))

#HISTOGRAM FEATURES
print("\nevaluating histogram accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))

#HU MOMENTS FEATURES
print("\nevaluating Hu Moments accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainHM, trainLabelsHM)
acc = model.score(testHM, testLabelsHM)
print("Hu Moments accuracy: {:.2f}%".format(acc * 100))

#----------------NAIVE BAYES CLASSIFICATION----------------
print("\n\n-------------------NAIVE BAYES CLASSIFICATION-------------------")
#RAW PIXEL FEATURES
print("evaluating raw pixel accuracy:")
nb_classifier_for_raw_pixels = GaussianNB()
nb_classifier_for_raw_pixels.fit(trainRI, trainRL)
y_pred = nb_classifier_for_raw_pixels.predict(testRI)
print("Accuracy:", accuracy_score(testRL, y_pred))
print(classification_report(testRL, y_pred))

#HISTOGRAM FEATURES
print("evaluating histogram accuracy:")
nb_classifier_for_histograms = GaussianNB()
nb_classifier_for_histograms.fit(trainFeat, trainLabels)
y_pred = nb_classifier_for_histograms.predict(testFeat)
print("Accuracy:", accuracy_score(testLabels, y_pred))
print(classification_report(testLabels, y_pred))

#HU MOMENTS FEATURES
print("evaluating Hu Moments accuracy:")	
nb_classifier_for_hu_moments = GaussianNB()
nb_classifier_for_hu_moments.fit(trainHM, trainLabelsHM)
y_pred = nb_classifier_for_hu_moments.predict(testHM)
print("Accuracy:", accuracy_score(testLabelsHM, y_pred))
print(classification_report(testLabelsHM, y_pred))


#----------------SVM CLASSIFICATION----------------
print("\n\n-------------------SVMCLASSIFICATION-------------------")
#RAW PIXEL FEATURES
print("evaluating SVM accuracy using raw pixel features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainRI, trainRL)
pipe.score(testRI, testRL)
print(classification_report(testRL, pipe.predict(testRI)))

#HISTOGRAM FEATURES
print("\nevaluating SVM accuracy using histogram features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainFeat, trainLabels)
pipe.score(testFeat, testLabels)
print(classification_report(testLabels, pipe.predict(testFeat)))

#HU MOMENTS FEATURES
print("\nevaluating SVM accuracy using Hu Moments features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainHM, trainLabelsHM)
pipe.score(testHM, testLabelsHM)
print(classification_report(testLabelsHM, pipe.predict(testHM)))