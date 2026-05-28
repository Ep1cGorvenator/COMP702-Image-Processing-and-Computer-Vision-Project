import cv2
import os
import numpy as np
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────

# __file__ is the path of THIS script (augmentation.py)
# We go up one level from src/ to reach the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Now build paths relative to the project root
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# Define where processed images will go
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# Define specific paths for training and testing splits
TRAIN_DIR     = os.path.join(PROCESSED_DIR, "train")
TEST_CLEAN    = os.path.join(PROCESSED_DIR, "test", "clean")
TEST_AUG      = os.path.join(PROCESSED_DIR, "test", "augmented")

# ── Random Seed ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Configuration ──────────────────────────────────────────────────────────

DENOMINATIONS = ["R10", "R20", "R50", "R100", "R200"]
DESIGNS       = ["old", "new"]
SIDES         = ["front", "back"]

# How many images to hold back per design type
TEST_HOLD_BACK = {
    "old": 2,   # hold back 2 of 8
    "new": 3    # hold back 3 of 12
}

# ── Helper Functions ───────────────────────────────────────────────────────

# Helper function to save augmented images with descriptive filenames in the processed/ folder
def save_image(image, output_dir, filename, suffix=None):
    """
    Saves an image to output_dir.
    If suffix provided, appends it to filename.
    If no suffix, saves with original filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    if suffix:
        name, ext = os.path.splitext(filename)
        filename  = f"{name}_{suffix}{ext}"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)

# ── Transformation Functions ───────────────────────────────────────────────

def rotate_image(image, angle):
    """
    Rotates an image by a given angle.
    Expands the canvas so the full note remains visible.
    """
    height, width = image.shape[:2]

    # Find the centre point of the image
    centre = (width // 2, height // 2)

    # Get the rotation matrix
    # This tells OpenCV how to rotate every pixel
    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    # Calculate the new canvas size after rotation
    # so corners don't get cut off
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    new_width  = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to account for the new canvas size
    rotation_matrix[0, 2] += (new_width / 2) - centre[0]
    rotation_matrix[1, 2] += (new_height / 2) - centre[1]

    # Apply the rotation with a white background
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        borderValue=(255, 255, 255)  # white background
    )

    return rotated

def flip_horizontal(image):
    """Mirrors the note left to right."""
    return cv2.flip(image, 1)

def flip_vertical(image):
    """Flips the note upside down."""
    return cv2.flip(image, 0)

def scale_image(image, factor):
    """
    Resizes the image by a scale factor.
    factor < 1.0 = smaller (far away)
    factor > 1.0 = larger  (close up)
    """
    height, width = image.shape[:2]
    new_width  = int(width * factor)
    new_height = int(height * factor)
    return cv2.resize(image, (new_width, new_height))

def adjust_brightness(image, factor=None, mode='up'):
    """
    Adjusts brightness with optional randomisation.
    mode='up'   → brighter, factor range 1.2–1.8
    mode='down' → darker,   factor range 0.3–0.7
    """
    if factor is None:
        factor = np.random.uniform(1.2, 1.8) if mode == 'up' \
            else np.random.uniform(0.3, 0.7)

    hsv          = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float64')
    hsv[:, :, 2] = (hsv[:, :, 2] * factor).clip(0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

def add_gaussian_noise(image, intensity=None):
    """
    Adds random pixel noise with randomised intensity.
    intensity range 15–35 simulates realistic camera grain.
    """
    if intensity is None:
        intensity = np.random.uniform(15, 35)

    noise = np.random.normal(0, intensity, image.shape).astype('int16')
    return (image.astype('int16') + noise).clip(0, 255).astype('uint8')

def apply_blur(image, kernel_size=3):
    """
    Applies Gaussian blur to simulate slight
    camera shake or out of focus photography.
    kernel_size must be an odd number.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def perspective_warp(image, strength=None):
    """
    Applies a randomised perspective transformation.
    Both the strength and corner positions are randomised
    to simulate realistic hand-held photography variation.
    
    strength=None means auto-randomise within a sensible range.
    strength can be set manually for testing purposes.
    """
    height, width = image.shape[:2]

    # Auto-randomise strength if not manually specified
    # Range 10-45px gives subtle to moderate warps
    # Avoids extremes that look unrealistic
    if strength is None:
        strength = np.random.uniform(10, 45)

    src_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    dst_points = np.float32([
        [np.random.uniform(0, strength),
         np.random.uniform(0, strength)],
        [np.random.uniform(width - strength, width),
         np.random.uniform(0, strength)],
        [np.random.uniform(0, strength),
         np.random.uniform(height - strength, height)],
        [np.random.uniform(width - strength, width),
         np.random.uniform(height - strength, height)]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(
        image, matrix, (width, height),
        borderValue=(255, 255, 255)
    )

# ── Augmentation Set ───────────────────────────────────────────────────────

def get_augmentations(image):
    """
    Returns all 16 augmented variants of a single image.
    Each entry is (transformed_image, suffix).
    """
    return [
        (rotate_image(image, 45),            "rot45"),
        (rotate_image(image, 90),            "rot90"),
        (rotate_image(image, 135),           "rot135"),
        (rotate_image(image, 180),           "rot180"),
        (rotate_image(image, 270),           "rot270"),
        (flip_horizontal(image),             "fliph"),
        (flip_vertical(image),               "flipv"),
        (scale_image(image, 0.75),           "scale75"),
        (scale_image(image, 0.50),           "scale50"),
        (scale_image(image, 1.25),           "scale125"),
        (scale_image(image, 1.50),           "scale150"),
        (adjust_brightness(image, mode='up'),   "bright_up"),
        (adjust_brightness(image, mode='down'), "bright_down"),
        (add_gaussian_noise(image),          "noise"),
        (apply_blur(image),                  "blur"),
        (perspective_warp(image),            "warp"),
    ]


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_augmentation_pipeline():
    """
    Walks through every image in dataset/raw/.
    Randomly splits into train and test using RANDOM_SEED.

    Train images:
        → Original copied to processed/train/
        → 16 augmented variants saved to processed/train/

    Test images:
        → Original copied to processed/test/clean/
        → 16 augmented variants saved to processed/test/augmented/
    """

    total_train_original  = 0
    total_train_augmented = 0
    total_test_clean      = 0
    total_test_augmented  = 0

    for denomination in DENOMINATIONS:
        for design in DESIGNS:
            for side in SIDES:

                # ── Input and output paths ─────────────────────────────────

                input_dir   = os.path.join(RAW_DIR, denomination, design, side)

                train_out   = os.path.join(TRAIN_DIR, denomination, design, side)
                clean_out   = os.path.join(TEST_CLEAN, denomination, design, side)
                aug_out     = os.path.join(TEST_AUG,   denomination, design, side)

                # ── Load and sort images ───────────────────────────────────

                all_images = sorted([
                    f for f in os.listdir(input_dir)
                    if f.lower().endswith('.jpg')
                ])

                if not all_images:
                    print(f"WARNING: No images found in {input_dir}")
                    continue

                # ── Random train/test split ────────────────────────────────

                hold_back   = TEST_HOLD_BACK[design]

                # Shuffle using our fixed seed for reproducibility
                indices     = list(range(len(all_images)))
                np.random.shuffle(indices)

                test_indices  = indices[:hold_back]
                train_indices = indices[hold_back:]

                train_images = [all_images[i] for i in sorted(train_indices)]
                test_images  = [all_images[i] for i in sorted(test_indices)]

                print(f"\n── {denomination} | {design} | {side} ──")
                print(f"   Train: {len(train_images)} | Test: {len(test_images)}")

                # ── Process training images ────────────────────────────────

                for filename in train_images:
                    image_path = os.path.join(input_dir, filename)
                    image      = cv2.imread(image_path)

                    if image is None:
                        print(f"   ERROR reading: {filename}")
                        continue

                    # Copy original to train
                    save_image(image, train_out, filename)
                    total_train_original += 1

                    # Generate and save 16 augmented variants
                    for augmented, suffix in get_augmentations(image):
                        save_image(augmented, train_out, filename, suffix)
                        total_train_augmented += 1

                # ── Process test images ────────────────────────────────────

                for filename in test_images:
                    image_path = os.path.join(input_dir, filename)
                    image      = cv2.imread(image_path)

                    if image is None:
                        print(f"   ERROR reading: {filename}")
                        continue

                    # Copy original to test/clean
                    save_image(image, clean_out, filename)
                    total_test_clean += 1

                    # Generate and save 16 augmented variants to test/augmented
                    for augmented, suffix in get_augmentations(image):
                        save_image(augmented, aug_out, filename, suffix)
                        total_test_augmented += 1

    # ── Final Summary ──────────────────────────────────────────────────────

    print(f"\n{'─' * 55}")
    print(f"  Augmentation Pipeline Complete")
    print(f"{'─' * 55}")
    print(f"  Random seed used:           {RANDOM_SEED}")
    print(f"{'─' * 55}")
    print(f"  Training originals:         {total_train_original}")
    print(f"  Training augmented:         {total_train_augmented}")
    print(f"  Training total:             {total_train_original + total_train_augmented}")
    print(f"{'─' * 55}")
    print(f"  Test clean (Set A):         {total_test_clean}")
    print(f"  Test augmented (Set B):     {total_test_augmented}")
    print(f"  Test total:                 {total_test_clean + total_test_augmented}")
    print(f"{'─' * 55}")
    print(f"  Grand total images:         "
          f"{total_train_original + total_train_augmented + total_test_clean + total_test_augmented}")
    print(f"{'─' * 55}")


# ── Run ────────────────────────────────────────────────────────────────────

run_augmentation_pipeline()