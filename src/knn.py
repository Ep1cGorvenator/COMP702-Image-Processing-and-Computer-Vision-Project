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

# grab the list of images that we'll be describing
print("[INFO] describing images...")
BASE_DIR = Path(__file__).resolve().parent.parent
imagePaths = list(paths.list_images(BASE_DIR / "dataset" / "processed"))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []
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
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
		
# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

#---------TRAIN TEST SPLIT----------------

# partition the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.20, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.20, random_state=42)

#----------------KNN CLASSIFICATION----------------
print("-------------------KNN CLASSIFICATION-------------------")
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

#----------------NAIVE BAYES CLASSIFICATION----------------
print("\n\n\n-------------------NAIVE BAYES CLASSIFICATION-------------------")
nb_classifier = GaussianNB()
nb_classifier.fit(trainRI, trainRL)

y_pred = nb_classifier.predict(testRI)

print("Accuracy:", accuracy_score(testRL, y_pred))
print(classification_report(testRL, y_pred))

print("Labels:", np.unique(labels))

#----------------SVM CLASSIFICATION----------------
print("\n\n\n-------------------SVMCLASSIFICATION-------------------")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainRI, trainRL)
pipe.score(testRI, testRL)
print(classification_report(testRL, pipe.predict(testRI)))