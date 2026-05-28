import cv2
import imutils
from pathlib import Path
from imutils import paths

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

def get_image_paths():
      BASE_DIR = Path(__file__).resolve().parent.parent
      return list(paths.list_images(BASE_DIR / "dataset" / "processed"))
  
def get_training_data():
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
            	
    return rawImages, features, labels