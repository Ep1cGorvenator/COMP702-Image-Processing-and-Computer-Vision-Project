import numpy as np
import cv2

from imutils import paths

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

#image_paths = list(paths.list_images('dataset/processed/'))

#print(f'opencv version: {cv2.__version__}')

#d_sample_count = 100
#image_count = len(image_paths)
#segmented_feature_vectors = np.zeros((image_count * d_sample_count, 128))
#final_feature_vectors = []
#vocabulary_size = 5

def sift_train(training_image_paths, d_sample_count, vocabulary_size):
    num_images = len(training_image_paths)
    segmented_feature_vectors = np.zeros((num_images * d_sample_count, 128))
    final_feature_vectors = []

    for path_idx in range(num_images):
        image = cv2.imread(training_image_paths[path_idx])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        descriptor_samples = descriptors[np.random.randint(descriptors.shape[0], size=d_sample_count)]

        segmented_feature_vectors[(path_idx + d_sample_count):(path_idx + 2*d_sample_count),] = descriptor_samples[:]
        print(segmented_feature_vectors)

    vocabulary = KMeans(n_clusters=vocabulary_size, random_state=0, n_init='auto').fit(segmented_feature_vectors)

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

        final_feature_vectors.append(image_features)

    final_feature_vectors = np.asarray(final_feature_vectors)

    features_norm_div = np.linalg.norm(final_feature_vectors, axis=1)
    for i in range(final_feature_vectors.shape[0]):
        final_feature_vectors[i] = final_feature_vectors[i] / features_norm_div[i]

    return final_feature_vectors, vocabulary

def sift_test(testing_image_paths, vocabulary, vocabulary_size):
    num_images = len(testing_image_paths)
    final_feature_vectors = []

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

        final_feature_vectors.append(image_features)

    final_feature_vectors = np.asarray(final_feature_vectors)

    features_norm_div = np.linalg.norm(final_feature_vectors, axis=1)
    for i in range(final_feature_vectors.shape[0]):
        final_feature_vectors[i] = final_feature_vectors[i] / features_norm_div[i]

    return final_feature_vectors
