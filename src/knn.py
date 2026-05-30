# import the necessary packages
from sklearn import tree
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

import sift_features as sf

import Haarlick_Features as haar

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

# ── COMMENTED OUT — replaced with directory-based split ───────────────────
# Original approach loaded all images and used sklearn train_test_split
# Replaced to respect our predefined train/test directory structure

# # grab the list of images that we'll be describing
# print("[INFO] describing images...")
# BASE_DIR = Path(__file__).resolve().parent.parent
# imagePaths = list(paths.list_images(BASE_DIR / "dataset" / "preprocessed"))

# # initialize the raw pixel intensities matrix, the histogram features matrix, and HU Moments features matrix
# rawImages = []
# features = []
# labels = []
# HUMoments = []

# #----------------FEATURE EXTRACTION----------------
# # loop over the input images
# for (i, imagePath) in enumerate(imagePaths):
# 	image = cv2.imread(imagePath)
# 	image_file = os.path.basename(imagePath)
	
# 	if "R100" in image_file:
# 		label = "R100"
# 	elif "R200" in image_file:	
# 		label = "R200"
# 	elif "R50" in image_file:
# 		label = "R50"
# 	elif "R10" in image_file:
# 		label = "R10"	
# 	elif "R20" in image_file:
# 		label = "R20"
# 	# extract raw pixel intensity "features"
# 	pixels = image_to_feature_vector(image)
# 	# extract color histogram to characterize the color distribution of the pixels in the image
# 	hist = extract_color_histogram(image)

# 	# update the raw images, features, and labels matricies,
# 	# respectively
# 	rawImages.append(pixels)
# 	features.append(hist)
# 	HUMoments.append(extract_hu_moments(image))

# 	labels.append(label)
# 	# show an update every 1,000 images
# 	if i > 0 and i % 1000 == 0:
# 		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
	
# # show some information on the memory consumed by the raw images
# # matrix and features matrix
# rawImages = np.array(rawImages)
# features = np.array(features)
# HUMoments = np.array(HUMoments)
# labels = np.array(labels)

# print("\nHow much memory is being consumed by the raw images matrix and features matrix:")
# print("pixels matrix: {:.2f}MB".format(
# 	rawImages.nbytes / (1024 * 1000.0)))
# print("features matrix: {:.2f}MB".format(
# 	features.nbytes / (1024 * 1000.0)))
# print("Hu Moments matrix: {:.2f}MB".format(
# 	HUMoments.nbytes / (1024 * 1000.0)))

# #---------TRAIN TEST SPLIT----------------
# #Using raw pixel intensities as features for training and testing
# (trainRI, testRI, trainRL, testRL) = train_test_split(
# 	rawImages, labels, test_size=0.20, random_state=42)

# #Using color histograms as features for training and testing
# (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
# 	features, labels, test_size=0.20, random_state=42)

# #Using HU Moments as features for training and testing
# (trainHM, testHM, trainLabelsHM, testLabelsHM) = train_test_split(
# 	HUMoments, labels, test_size=0.20, random_state=42)

# #SIFT setup
# (trainImagePaths, testImagePaths, siftTrainLabels, siftTestLabels) = train_test_split(
# 	imagePaths, labels, test_size=0.2, random_state=42)
# sift_x_train, sift_x_test = sf.sift_extract(trainImagePaths, testImagePaths)

# #Haarlick features
# (trainImgPaths, testImgPaths , trainLabelsHaar, testLabelsHaar) = train_test_split(
# 	imagePaths, labels, test_size=0.2, random_state=42)

# haarlick_train= haar.haarlickTrain(trainImgPaths)
# haarlick_test = haar.haarlickTest(testImgPaths)

# ── Directory-Based Train/Test Split ───────────────────────────────────────
# Loads images from our predefined directory structure
# Respects the train/test split defined during augmentation
# Training: dataset/preprocessed/train/
# Testing:  dataset/preprocessed/test/clean/

BASE_DIR     = Path(__file__).resolve().parent.parent
PREPROC_BASE = BASE_DIR / "dataset" / "preprocessed"
TRAIN_PATH   = PREPROC_BASE / "train"
TEST_PATH    = PREPROC_BASE / "test" / "clean"

# ── Label Extraction ───────────────────────────────────────────────────────

def get_label(image_path):
    """
    Extracts denomination label from filename.
    Order matters — check R100/R200 before R10/R20
    to avoid partial string matches.
    R100 contains R10 as substring — must check longer first.
    """
    filename = os.path.basename(str(image_path))
    if "R200" in filename:   return "R200"
    elif "R100" in filename: return "R100"
    elif "R50"  in filename: return "R50"
    elif "R20"  in filename: return "R20"
    elif "R10"  in filename: return "R10"
    return None

# ── Scan Directories ───────────────────────────────────────────────────────

print("[INFO] Scanning directories...")

trainImagePaths = [str(p) for p in paths.list_images(str(TRAIN_PATH))]
testImagePaths  = [str(p) for p in paths.list_images(str(TEST_PATH))]

print(f"[INFO] Training images found: {len(trainImagePaths)}")
print(f"[INFO] Test images found:     {len(testImagePaths)}")

# ── Verify Labels ──────────────────────────────────────────────────────────

# Extract labels for all paths
# Verify no None labels — would indicate naming issue
train_label_check = [get_label(p) for p in trainImagePaths]
test_label_check  = [get_label(p) for p in testImagePaths]

none_train = train_label_check.count(None)
none_test  = test_label_check.count(None)

if none_train > 0:
    print(f"[WARNING] {none_train} training images could not be labelled")
if none_test > 0:
    print(f"[WARNING] {none_test} test images could not be labelled")

print(f"\n[INFO] Label distribution — Training:")
for denomination in ["R10", "R20", "R50", "R100", "R200"]:
    count = train_label_check.count(denomination)
    print(f"  {denomination}: {count} images")

print(f"\n[INFO] Label distribution — Test:")
for denomination in ["R10", "R20", "R50", "R100", "R200"]:
    count = test_label_check.count(denomination)
    print(f"  {denomination}: {count} images")
    
# ── Quick Verification Before Full Feature Extraction ─────────────────────

print(f"\n{'─' * 55}")
print(f"  Directory Split Verification")
print(f"{'─' * 55}")
print(f"  Training images:  {len(trainImagePaths)}")
print(f"  Test images:      {len(testImagePaths)}")
print(f"  Total:            {len(trainImagePaths) + len(testImagePaths)}")
print(f"{'─' * 55}")

# Temporarily stop here to verify counts before running full extraction
# Comment out the line below once verified
import sys
# ── Feature Extraction ─────────────────────────────────────────────────────
# sys.exit("[INFO] Verification complete — comment out sys.exit() to run full pipeline")

# ── Hu Moments ─────────────────────────────────────────────────────────────

print("\n[INFO] Extracting Hu Moments — Training...")
trainHM         = []
trainLabelsHM   = []

for (i, image_path) in enumerate(trainImagePaths):
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    label = get_label(image_path)
    if label is None:
        continue
    trainHM.append(extract_hu_moments(image))
    trainLabelsHM.append(label)
    if i > 0 and i % 500 == 0:
        print(f"[INFO] training Hu Moments: {i}/{len(trainImagePaths)}")

print("[INFO] Extracting Hu Moments — Test...")
testHM          = []
testLabelsHM    = []

for (i, image_path) in enumerate(testImagePaths):
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    label = get_label(image_path)
    if label is None:
        continue
    testHM.append(extract_hu_moments(image))
    testLabelsHM.append(label)

trainHM       = np.array(trainHM)
testHM        = np.array(testHM)
trainLabelsHM = np.array(trainLabelsHM)
testLabelsHM  = np.array(testLabelsHM)

print(f"[INFO] Hu Moments — Train: {trainHM.shape} | Test: {testHM.shape}")

# ── SIFT Features ──────────────────────────────────────────────────────────

print("\n[INFO] Extracting SIFT features...")
siftTrainLabels = np.array([get_label(p) for p in trainImagePaths])
siftTestLabels  = np.array([get_label(p) for p in testImagePaths])

sift_x_train, sift_x_test = sf.sift_extract(
    trainImagePaths,
    testImagePaths
)
print("[INFO] SIFT extraction complete")

# ── Haralick Features ──────────────────────────────────────────────────────

print("\n[INFO] Extracting Haralick features...")
trainLabelsHaar = np.array([get_label(p) for p in trainImagePaths])
testLabelsHaar  = np.array([get_label(p) for p in testImagePaths])

haarlick_train  = haar.haarlickTrain(trainImagePaths)
haarlick_test   = haar.haarlickTest(testImagePaths)
print("[INFO] Haralick extraction complete")

# ── Memory Report ──────────────────────────────────────────────────────────

print(f"\n{'─' * 55}")
print(f"  Feature Extraction Complete")
print(f"{'─' * 55}")
print(f"  Hu Moments train:    {trainHM.shape}")
print(f"  Hu Moments test:     {testHM.shape}")
print(f"  SIFT train:          {sift_x_train.shape}")
print(f"  SIFT test:           {sift_x_test.shape}")
print(f"  Haralick train:      {len(haarlick_train)}")
print(f"  Haralick test:       {len(haarlick_test)}")
print(f"{'─' * 55}")

print("done")

#----------------KNN CLASSIFICATION----------------
print("\n-------------------KNN CLASSIFICATION-------------------")
#YOU CAN SPECIFY HOW MANY NEIGHBOURS TO USE WITH THE n_neighbors PARAMETER, 
# AND HOW MANY CPU CORES TO USE WITH THE n_jobs PARAMETER

# #RAW PIXEL FEATURES — commented out, variables undefined
# print("\nevaluating raw pixel accuracy:")
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
# model.fit(trainRI, trainRL)
# acc = model.score(testRI, testRL)
# print("raw pixel accuracy: {:.2f}%".format(acc * 100))

# #HISTOGRAM FEATURES — commented out, variables undefined
# print("\nevaluating histogram accuracy:")
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("histogram accuracy: {:.2f}%".format(acc * 100))

#HU MOMENTS FEATURES
print("\nevaluating Hu Moments accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainHM, trainLabelsHM)
acc = model.score(testHM, testLabelsHM)
print("Hu Moments accuracy: {:.2f}%".format(acc * 100))

#SIFT
print("\nevaluating SIFT accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(sift_x_train, siftTrainLabels)
acc = model.score(sift_x_test, siftTestLabels)
print("SIFT accuracy: {:.2f}%".format(acc * 100))

#HAARLICK FEATURES
print("\nevaluating Haarlick features accuracy:")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(haarlick_train, trainLabelsHaar)
acc = model.score(haarlick_test, testLabelsHaar)
print("Haarlick accuracy: {:.2f}%".format(acc * 100))

#----------------NAIVE BAYES CLASSIFICATION----------------
print("\n\n-------------------NAIVE BAYES CLASSIFICATION-------------------\n")
#RAW PIXEL FEATURES — commented out, variables undefined
# print("evaluating raw pixel accuracy:")
# nb_classifier_for_raw_pixels = GaussianNB()
# nb_classifier_for_raw_pixels.fit(trainRI, trainRL)
# y_pred = nb_classifier_for_raw_pixels.predict(testRI)
# print("raw pixel accuracy: {:.2f}%".format(nb_classifier_for_raw_pixels.score(testRI, testRL) * 100))

#HISTOGRAM FEATURES — commented out, variables undefined
# print("\nevaluating histogram accuracy:")
# nb_classifier_for_histograms = GaussianNB()
# nb_classifier_for_histograms.fit(trainFeat, trainLabels)
# y_pred = nb_classifier_for_histograms.predict(testFeat)
# print("histogram accuracy: {:.2f}%".format(nb_classifier_for_histograms.score(testFeat, testLabels) * 100))

#HU MOMENTS FEATURES
print("\nevaluating Hu Moments accuracy:")	
nb_classifier_for_hu_moments = GaussianNB()
nb_classifier_for_hu_moments.fit(trainHM, trainLabelsHM)
y_pred = nb_classifier_for_hu_moments.predict(testHM)
print("Hu Moments accuracy: {:.2f}%".format(nb_classifier_for_hu_moments.score(testHM, testLabelsHM) * 100))

#SIFT
print("\nevaluating SIFT accuracy:")
nb_classifier_for_sift = GaussianNB()
nb_classifier_for_sift.fit(sift_x_train, siftTrainLabels)
y_pred = nb_classifier_for_sift.predict(sift_x_test)
print("SIFT accuracy: {:.2f}%".format(nb_classifier_for_sift.score(sift_x_test, siftTestLabels) * 100))

#HAARLICK FEATURES
print("\nevaluating Haarlick features accuracy:")
nb_classifier_for_haarlick = GaussianNB()
nb_classifier_for_haarlick.fit(haarlick_train, trainLabelsHaar)
y_pred = nb_classifier_for_haarlick.predict(haarlick_test)
print("Haarlick accuracy: {:.2f}%".format(nb_classifier_for_haarlick.score(haarlick_test, testLabelsHaar) * 100))

#----------------SVM CLASSIFICATION----------------
print("\n\n-------------------SVM CLASSIFICATION-------------------\n")
#RAW PIXEL FEATURES — commented out, variables undefined
# print("evaluating SVM accuracy using raw pixel features:")
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(trainRI, trainRL)
# pipe.score(testRI, testRL)
# print("SVM accuracy using raw pixel features: {:.2f}%".format(pipe.score(testRI, testRL) * 100))

#HISTOGRAM FEATURES — commented out, variables undefined
# print("\nevaluating SVM accuracy using histogram features:")
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(trainFeat, trainLabels)
# pipe.score(testFeat, testLabels)
# print("SVM accuracy using histogram features: {:.2f}%".format(pipe.score(testFeat, testLabels) * 100))

#HU MOMENTS FEATURES
print("\nevaluating SVM accuracy using Hu Moments features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainHM, trainLabelsHM)
pipe.score(testHM, testLabelsHM)
print("SVM accuracy using Hu Moments features: {:.2f}%".format(pipe.score(testHM, testLabelsHM) * 100))

#SIFT
print("\nevaluating SVM accuracy using sift features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(sift_x_train, siftTrainLabels)
pipe.score(sift_x_test, siftTestLabels)
print("SVM accuracy using sift features: {:.2f}%".format(pipe.score(sift_x_test,siftTestLabels) * 100))

#HAARLICK FEATURES
print("\nevaluating SVM accuracy using Haarlick features:")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(haarlick_train, trainLabelsHaar)
pipe.score(haarlick_test, testLabelsHaar)
print("SVM accuracy using Haarlick features: {:.2f}%".format(pipe.score(haarlick_test, testLabelsHaar) * 100))

#----------------DECISION TREE CLASSIFICATION----------------
clf = tree.DecisionTreeClassifier()
print("\n\n-------------------DECISION TREE CLASSIFICATION-------------------")
#RAW PIXEL FEATURES — commented out, variables undefined
# print("\nevaluating Decision Tree accuracy using raw pixel features:")
# clf.fit(trainRI, trainRL)
# y_pred = clf.predict(testRI)
# print("raw pixel Accuracy: {:.2f}%".format(accuracy_score(testRL, y_pred) * 100))

#HISTOGRAM FEATURES — commented out, variables undefined
# print("\nevaluating Decision Tree accuracy using histogram features:")
# clf.fit(trainFeat, trainLabels)
# y_pred = clf.predict(testFeat)
# print("Histogram features Accuracy: {:.2f}%".format(accuracy_score(testLabels, y_pred) * 100))

#HU MOMENTS FEATURES
print("\nevaluating Decision Tree accuracy using Hu Moments features:")
clf.fit(trainHM, trainLabelsHM)
y_pred = clf.predict(testHM)
print("HU Moments Accuracy: {:.2f}%".format(accuracy_score(testLabelsHM, y_pred) * 100))

#SIFT
print("\nevaluating Decision Tree accuracy using sift features:")
clf.fit(sift_x_train, siftTrainLabels)
y_pred = clf.predict(sift_x_test)
print("SIFT Accuracy: {:.2f}%".format(accuracy_score(siftTestLabels, y_pred) * 100))

#HARALICK
print("\nevaluating Decision Tree accuracy using Haralick features:")
clf.fit(haarlick_train, trainLabelsHaar)
y_pred = clf.predict(haarlick_test)
print("Haarlick Accuracy: {:.2f}%".format(accuracy_score(testLabelsHaar, y_pred) * 100))