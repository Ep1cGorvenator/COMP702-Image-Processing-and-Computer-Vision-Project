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

from ImageProcessing import get_training_data


test_rawImages, test_features, test_labels = get_training_data()

rawImages = np.array(test_rawImages)
features = np.array(test_features)
labels = np.array(test_labels)
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
print("\n\n\n-------------------KNN CLASSIFICATION-------------------")
print("[INFO] evaluating raw pixel accuracy...")
#YOU CAN SPECIFY HOW MANY NEIGHBOURS TO USE WITH THE n_neighbors PARAMETER, 
# AND HOW MANY CPU CORES TO USE WITH THE n_jobs PARAMETER
model = KNeighborsClassifier(n_neighbors=1,n_jobs=4)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
#YOU CAN SPECIFY HOW MANY NEIGHBOURS TO USE WITH THE n_neighbors PARAMETER, 
# AND HOW MANY CPU CORES TO USE WITH THE n_jobs PARAMETER
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
print("\n\n\n-------------------SVM CLASSIFICATION-------------------")
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(trainRI, trainRL)
pipe.score(testRI, testRL)
print(classification_report(testRL, pipe.predict(testRI)))