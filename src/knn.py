# import the necessary packages
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils
import cv2
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathlib import Path

import sift_features as sf

import orb_features as of

import Haarlick_Features as haar

import hu_moments_features as hu

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


BASE_DIR     = Path(__file__).resolve().parent.parent
PREPROC_BASE = BASE_DIR / "dataset" / "preprocessed"
TRAIN_PATH   = PREPROC_BASE / "train"
TEST_PATH    = PREPROC_BASE / "test" 

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

# ── Feature Extraction ─────────────────────────────────────────────────────

# ── Hu Moments ─────────────────────────────────────────────────────────────

#USING ADAPTIVE THRESHOLDING
print("\n[INFO] Extracting HU Moment features using adaptive thresholding...")
trainLabelsHuMoments = np.array([get_label(p) for p in trainImagePaths])
testLabelsHuMoments  = np.array([get_label(p) for p in testImagePaths])

huMoments_adaptive_train  = hu.huMomentTrainAdaptThresh(trainImagePaths)
huMoments_adaptive_test   = hu.huMomentTestAdaptThresh(testImagePaths)
print("[INFO] HU Moment extraction using adaptive thresholding complete")

#USING OTSU THRESHOLDING
print("\n[INFO] Extracting HU Moment features using OTSU thresholding...")

huMoments_otsu_train  = hu.huMomentTrainOtsuThresh(trainImagePaths)
huMoments_otsu_test   = hu.huMomentTestOtsuThresh(testImagePaths)
print("[INFO] HU Moment extraction using OTSU thresholding complete")

#USING NO THRESHOLDING
print("\n[INFO] Extracting HU Moment features using OTSU thresholding...")

huMoments_train = hu.huMomentTrain(trainImagePaths)
huMoments_test = hu.huMomentTest(testImagePaths)
print("[INFO] HU Moment extraction using OTSU thresholding complete")

# # ── SIFT Features ──────────────────────────────────────────────────────────

# print("\n[INFO] Extracting SIFT features...")
# siftTrainLabels = np.array([get_label(p) for p in trainImagePaths])
# siftTestLabels  = np.array([get_label(p) for p in testImagePaths])

# sift_x_train, sift_x_test = sf.sift_extract(
#     trainImagePaths,
#     testImagePaths
# )
# print("[INFO] SIFT extraction complete")

# # ── ORB Features ───────────────────────────────────────────────────────────

# print("\n[INFO] Extracting ORB features...")
# orbTrainLabels = np.array([get_label(p) for p in trainImagePaths])
# orbTestLabels  = np.array([get_label(p) for p in testImagePaths])

# print("\n[INFO] Extracting ORB features...")
# orb_x_train, orb_x_test = of.orb_extract(
#     trainImagePaths,
#     testImagePaths
# )
# print("[INFO] ORB extraction complete")

# # ── Haralick Features ──────────────────────────────────────────────────────

# print("\n[INFO] Extracting Haralick features...")
# trainLabelsHaar = np.array([get_label(p) for p in trainImagePaths])
# testLabelsHaar  = np.array([get_label(p) for p in testImagePaths])

# haarlick_train  = haar.haarlickTrain(trainImagePaths)
# haarlick_test   = haar.haarlickTest(testImagePaths)
# print("[INFO] Haralick extraction complete")

# # ── Memory Report ──────────────────────────────────────────────────────────

# print(f"\n{'─' * 55}")
# print(f"  Feature Extraction Complete")
# print(f"{'─' * 55}")
# print(f"  Hu Moments train:    {len(huMoments_adaptive_train)}")
# print(f"  Hu Moments test:     {len(huMoments_adaptive_test)}")
# print(f"  SIFT train:          {sift_x_train.shape}")
# print(f"  SIFT test:           {sift_x_test.shape}")
# print(f"  Haralick train:      {len(haarlick_train)}")
# print(f"  Haralick test:       {len(haarlick_test)}")
# print(f"{'─' * 55}")

# print("done")

#----------------KNN CLASSIFICATION----------------
print("\n-------------------KNN CLASSIFICATION-------------------")
#YOU CAN SPECIFY HOW MANY NEIGHBOURS TO USE WITH THE n_neighbors PARAMETER, 
# AND HOW MANY CPU CORES TO USE WITH THE n_jobs PARAMETER

#HU MOMENTS FEATURES
print("\nevaluating Hu Moments accuracy (using adaptive thresholding):")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(huMoments_adaptive_train, trainLabelsHuMoments)
acc = model.score(huMoments_adaptive_test, testLabelsHuMoments)
print("Hu Moments (adaptive thresholding) accuracy: {:.2f}%".format(acc * 100))

print("\nevaluating Hu Moments accuracy (using otsu thresholding):")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(huMoments_otsu_train, trainLabelsHuMoments)
acc = model.score(huMoments_otsu_test, testLabelsHuMoments)
print("Hu Moments (otsu thresholding) accuracy: {:.2f}%".format(acc * 100))

print("\nevaluating Hu Moments accuracy (using no thresholding):")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(huMoments_train, trainLabelsHuMoments)
acc = model.score(huMoments_test, testLabelsHuMoments)
print("Hu Moments (no thresholding) accuracy: {:.2f}%".format(acc * 100))

# #SIFT
# print("\nevaluating SIFT accuracy:")
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
# model.fit(sift_x_train, siftTrainLabels)
# acc = model.score(sift_x_test, siftTestLabels)
# print("SIFT accuracy: {:.2f}%".format(acc * 100))

# #HAARLICK FEATURES
# print("\nevaluating Haarlick features accuracy:")
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
# model.fit(haarlick_train, trainLabelsHaar)
# acc = model.score(haarlick_test, testLabelsHaar)
# print("Haarlick accuracy: {:.2f}%".format(acc * 100))

# #ORB FEATURES
# print("\nevaluating ORB features accuracy:")
# model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
# model.fit(orb_x_train, orbTrainLabels) 
# acc = model.score(orb_x_test, orbTestLabels)  
# print("ORB accuracy: {:.2f}%".format(acc * 100))

#----------------NAIVE BAYES CLASSIFICATION----------------
print("\n\n-------------------NAIVE BAYES CLASSIFICATION-------------------\n")
#RAW PIXEL FEATURES — commented out, variables undefined

#HU MOMENTS FEATURES
print("\nevaluating Hu Moments accuracy using adaptive thresholding:")	
nb_classifier_for_hu_moments = GaussianNB()
nb_classifier_for_hu_moments.fit(huMoments_adaptive_train, trainLabelsHuMoments)
y_pred = nb_classifier_for_hu_moments.predict(huMoments_adaptive_test)
print("Hu Moments accuracy: {:.2f}%".format(nb_classifier_for_hu_moments.score(huMoments_adaptive_test, testLabelsHuMoments) * 100))

print("\nevaluating Hu Moments accuracy using otsu thresholding:")	
nb_classifier_for_hu_moments = GaussianNB()
nb_classifier_for_hu_moments.fit(huMoments_otsu_train, trainLabelsHuMoments)
y_pred = nb_classifier_for_hu_moments.predict(huMoments_otsu_test)
print("Hu Moments accuracy: {:.2f}%".format(nb_classifier_for_hu_moments.score(huMoments_otsu_test, testLabelsHuMoments) * 100))

print("\nevaluating Hu Moments accuracy using no thresholding:")    
nb_classifier_for_hu_moments = GaussianNB()
nb_classifier_for_hu_moments.fit(huMoments_train, trainLabelsHuMoments)
y_pred = nb_classifier_for_hu_moments.predict(huMoments_test)
print("Hu Moments accuracy: {:.2f}%".format(nb_classifier_for_hu_moments.score(huMoments_test, testLabelsHuMoments) * 100))

# #SIFT
# print("\nevaluating SIFT accuracy:")
# nb_classifier_for_sift = GaussianNB()
# nb_classifier_for_sift.fit(sift_x_train, siftTrainLabels)
# y_pred = nb_classifier_for_sift.predict(sift_x_test)
# print("SIFT accuracy: {:.2f}%".format(nb_classifier_for_sift.score(sift_x_test, siftTestLabels) * 100))

# #HAARLICK FEATURES
# print("\nevaluating Haarlick features accuracy:")
# nb_classifier_for_haarlick = GaussianNB()
# nb_classifier_for_haarlick.fit(haarlick_train, trainLabelsHaar)
# y_pred = nb_classifier_for_haarlick.predict(haarlick_test)
# print("Haarlick accuracy: {:.2f}%".format(nb_classifier_for_haarlick.score(haarlick_test, testLabelsHaar) * 100))

# #ORB FEATURES
# print("\nevaluating ORB features accuracy:")
# nb_classifier_for_orb = GaussianNB()
# nb_classifier_for_orb.fit(orb_x_train, orbTrainLabels)
# y_pred = nb_classifier_for_orb.predict(orb_x_test)
# print("ORB accuracy: {:.2f}%".format(nb_classifier_for_orb.score(orb_x_test, orbTestLabels) * 100))

#----------------SVM CLASSIFICATION----------------
print("\n\n-------------------SVM CLASSIFICATION-------------------\n")
#RAW PIXEL FEATURES — commented out, variables undefined

#HU MOMENTS FEATURES
print("\nevaluating SVM accuracy using Hu Moments features (adaptive thresholding):")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(huMoments_adaptive_train, trainLabelsHuMoments)
pipe.score(huMoments_adaptive_test, testLabelsHuMoments)
print("SVM accuracy using Hu Moments features: {:.2f}%".format(pipe.score(huMoments_adaptive_test, testLabelsHuMoments) * 100))

print("\nevaluating SVM accuracy using Hu Moments features (otsu thresholding):")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(huMoments_otsu_train, trainLabelsHuMoments)
pipe.score(huMoments_otsu_test, testLabelsHuMoments)
print("SVM accuracy using Hu Moments features: {:.2f}%".format(pipe.score(huMoments_otsu_test, testLabelsHuMoments) * 100))

print("\nevaluating SVM accuracy using Hu Moments features (no thresholding):")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(huMoments_train, trainLabelsHuMoments)
pipe.score(huMoments_test, testLabelsHuMoments)
print("SVM accuracy using Hu Moments features: {:.2f}%".format(pipe.score(huMoments_test, testLabelsHuMoments) * 100))

# #SIFT
# print("\nevaluating SVM accuracy using sift features:")
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(sift_x_train, siftTrainLabels)
# pipe.score(sift_x_test, siftTestLabels)
# print("SVM accuracy using sift features: {:.2f}%".format(pipe.score(sift_x_test,siftTestLabels) * 100))

# #HAARLICK FEATURES
# print("\nevaluating SVM accuracy using Haarlick features:")
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(haarlick_train, trainLabelsHaar)
# pipe.score(haarlick_test, testLabelsHaar)
# print("SVM accuracy using Haarlick features: {:.2f}%".format(pipe.score(haarlick_test, testLabelsHaar) * 100))

# #ORB FEATURES
# print("\nevaluating SVM accuracy using ORB features:")
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(orb_x_train, orbTrainLabels)
# pipe.score(orb_x_test, orbTestLabels)
# print("SVM accuracy using ORB features: {:.2f}%".format(pipe.score(orb_x_test, orbTestLabels) * 100))

#----------------DECISION TREE CLASSIFICATION----------------
clf = tree.DecisionTreeClassifier()
print("\n\n-------------------DECISION TREE CLASSIFICATION-------------------")

#HU MOMENTS FEATURES
print("\nevaluating Decision Tree accuracy using Hu Moments features (using adaptive thresholding):")
clf.fit(huMoments_adaptive_train, trainLabelsHuMoments)
y_pred = clf.predict(huMoments_adaptive_test)
print("HU Moments Accuracy: {:.2f}%".format(accuracy_score(testLabelsHuMoments, y_pred) * 100))

print("\nevaluating Decision Tree accuracy using Hu Moments features (using otsu thresholding):")
clf.fit(huMoments_otsu_train, trainLabelsHuMoments)
y_pred = clf.predict(huMoments_otsu_test)
print("HU Moments Accuracy: {:.2f}%".format(accuracy_score(testLabelsHuMoments, y_pred) * 100))

print("\nevaluating Decision Tree accuracy using Hu Moments features (using no thresholding):")
clf.fit(huMoments_train, trainLabelsHuMoments)
y_pred = clf.predict(huMoments_test)
print("HU Moments Accuracy: {:.2f}%".format(accuracy_score(testLabelsHuMoments, y_pred) * 100))


# #SIFT
# print("\nevaluating Decision Tree accuracy using sift features:")
# clf.fit(sift_x_train, siftTrainLabels)
# y_pred = clf.predict(sift_x_test)
# print("SIFT Accuracy: {:.2f}%".format(accuracy_score(siftTestLabels, y_pred) * 100))

# #HARALICK
# print("\nevaluating Decision Tree accuracy using Haralick features:")
# clf.fit(haarlick_train, trainLabelsHaar)
# y_pred = clf.predict(haarlick_test)
# print("Haarlick Accuracy: {:.2f}%".format(accuracy_score(testLabelsHaar, y_pred) * 100))

# #ORB FEATURES
# print("\nevaluating Decision Tree accuracy using ORB features:")
# clf.fit(orb_x_train, orbTrainLabels)
# y_pred = clf.predict(orb_x_test)
# print("ORB Accuracy: {:.2f}%".format(accuracy_score(orbTestLabels, y_pred) * 100))
