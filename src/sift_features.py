import threading
import copy

import numpy as np
import cv2

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

def sift_extract(training_image_paths, testing_image_paths, d_sample_count=100, vocabulary_size=200):
    print('vocabulary calculation...')
    vocabulary = sift_build_vocabulary(training_image_paths, d_sample_count, vocabulary_size)
    print('vocabulary calculation complete')

    num_train_images = len(training_image_paths)
    num_test_images = len(testing_image_paths)
    x_train = np.empty((num_train_images, vocabulary_size))
    x_test = np.empty((num_test_images, vocabulary_size))

    train_thread = threading.Thread(target=sift_train, args=(training_image_paths, vocabulary, x_train))
    test_thread = threading.Thread(target=sift_test, args=(testing_image_paths, vocabulary, x_test))

    train_thread.start()
    test_thread.start()

    train_thread.join()
    test_thread.join()

    return x_train, x_test

def sift_build_vocabulary(training_image_paths, d_sample_count, vocabulary_size):
    num_images = len(training_image_paths)
    segmented_feature_vectors = np.zeros((num_images * d_sample_count, 128))

    for path_idx in range(num_images):
        image = cv2.imread(training_image_paths[path_idx])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        descriptor_samples = descriptors[np.random.randint(descriptors.shape[0], size=d_sample_count)]

        segmented_feature_vectors[(path_idx + d_sample_count):(path_idx + 2*d_sample_count),] = descriptor_samples[:]

    vocabulary = KMeans(n_clusters=vocabulary_size, random_state=0, n_init='auto').fit(segmented_feature_vectors)

    return vocabulary


def sift_train(training_image_paths, vocabulary, target_feature_matrix):
    print('sift train...')
    num_images = len(training_image_paths)
    vocabulary_size = len(vocabulary.cluster_centers_)

    for path_idx in range(num_images):
        image = cv2.imread(training_image_paths[path_idx])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        distance = cdist(descriptors, vocabulary.cluster_centers_, 'euclidean')

        bin_assignment = np.argmin(distance, axis=1)

        image_features = np.zeros(vocabulary_size)
        for assignment_id in bin_assignment:
            image_features[assignment_id] += 1

        target_feature_matrix[path_idx] = image_features

    features_norm_div = np.linalg.norm(target_feature_matrix, axis=1)
    for i in range(target_feature_matrix.shape[0]):
        target_feature_matrix[i] = target_feature_matrix[i] / features_norm_div[i]

    print('sift train ended')

def sift_test(testing_image_paths, vocabulary, target_feature_matrix):
    print('sift test...')
    num_images = len(testing_image_paths)
    vocabulary_size = len(vocabulary.cluster_centers_)

    for path_idx in range(num_images):
        image = cv2.imread(testing_image_paths[path_idx])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        distance = cdist(descriptors, vocabulary.cluster_centers_, 'euclidean')

        bin_assignment = np.argmin(distance, axis=1)

        image_features = np.zeros(vocabulary_size)
        for assignment_id in bin_assignment:
            image_features[assignment_id] += 1

        target_feature_matrix[path_idx] = image_features


    features_norm_div = np.linalg.norm(target_feature_matrix, axis=1)
    for i in range(target_feature_matrix.shape[0]):
        target_feature_matrix[i] = target_feature_matrix[i] / features_norm_div[i]

    print('sift test ended')
