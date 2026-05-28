import cv2
import os
import numpy as np
import time

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
TRAIN_DIR     = os.path.join(PROCESSED_DIR, "train")
TEST_CLEAN    = os.path.join(PROCESSED_DIR, "test", "clean")
TEST_AUG      = os.path.join(PROCESSED_DIR, "test", "augmented")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "preprocessing")

# ── Configuration ──────────────────────────────────────────────────────────

TARGET_WIDTH  = 640
TARGET_HEIGHT = 320
DENOMINATIONS = ["R10", "R20", "R50", "R100", "R200"]
DESIGNS       = ["old", "new"]
SIDES         = ["front", "back"]

# ── Stage 1: Resizing ──────────────────────────────────────────────────────

def resize_image(image):
    """
    Resizes image to standard TARGET_WIDTH x TARGET_HEIGHT.
    Standardises all images regardless of source dimensions.
    Handles augmented variants with varying canvas sizes.
    """
    return cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

# ── Stage 2: Noise Reduction ───────────────────────────────────────────────

def apply_bilateral_filter(image):
    """
    Bilateral filter — selected noise reduction technique.
    Smooths uniform regions whilst preserving edges and fine texture.
    Selected over Gaussian and Median after empirical comparison.
    d          → diameter of pixel neighbourhood
    sigmaColor → how much colour difference is tolerated
    sigmaSpace → how much distance is tolerated
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# ── Stage 3: Contrast Enhancement ─────────────────────────────────────────

def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE — selected contrast enhancement technique.
    Selected over global histogram equalisation after comparison.
    Handles both BGR and grayscale input safely.
    clipLimit=2.0      → limits contrast amplification
    tileGridSize=(8,8) → balanced local tile enhancement
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )
    return clahe.apply(image)

# ── Stage 4: Colour Conversion ──────────────────────────────────────

def convert_to_grayscale(image):
    """
    Converts BGR image to single channel grayscale.
    Selected over HSV after reasoning:
    → Compatible with SIFT and ORB feature extractors
    → Retains structural and texture information
    → Robust to lighting variation
    → Avoids unnecessary feature vector complexity
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ── Stage 5: Normalisation ─────────────────────────────────────────────────

def normalise_image(image):
    """
    Scales pixel values from 0-255 to 0.0-1.0 range.
    Standardises inputs for SVM classifier.
    Returns float32 array.
    """
    return image.astype('float32') / 255.0

# ── Full Preprocessing Pipeline ────────────────────────────────────────────

def preprocess_image(image):
    """
    Applies all five preprocessing stages in sequence:
    1. Resize        → 640x320 standard dimensions
    2. Bilateral     → noise reduction, edge preserving
    3. Grayscale     → single channel conversion
    4. CLAHE         → local contrast enhancement
    5. Normalise     → pixel value standardisation
    Returns normalised float32 grayscale image.
    """
    resized    = resize_image(image)
    denoised   = apply_bilateral_filter(resized)
    gray       = convert_to_grayscale(denoised)
    enhanced   = apply_clahe(gray)
    normalised = normalise_image(enhanced)
    return normalised

# ── Quality Metrics ────────────────────────────────────────────────────────

def show_image(title, image, max_width=800):
    """
    Displays resized preview for screen viewing only.
    Does not affect image data.
    """
    h, w = image.shape[:2]
    if w > max_width:
        scale   = max_width / w
        preview = cv2.resize(image, (max_width, int(h * scale)))
    else:
        preview = image
    cv2.imshow(title, preview)


def calculate_l_std_dev(image):
    """
    Calculates standard deviation of L channel in LAB colour space.
    L = perceptual lightness (0=black, 100=white).
    Higher std dev = more contrast variation = better image quality.
    Accepts BGR or grayscale input.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return np.std(l)


def run_comparisons():
    """
    Runs L channel quality verification on sample images.
    Displays original vs preprocessed result visually.
    Prints before/after L std dev metrics.
    Use to verify preprocessing pipeline is working correctly.
    """
    sample_images = [
        os.path.join(TRAIN_DIR, "R20",  "old", "back",
                     "R20_old_back_001.jpg"),
        os.path.join(TRAIN_DIR, "R100", "new", "front",
                     "R100_new_front_002.jpg"),
        os.path.join(TRAIN_DIR, "R50",  "old", "front",
                     "R50_old_front_001.jpg"),
    ]

    print(f"\n{'─' * 55}")
    print(f"  Preprocessing Quality Verification")
    print(f"{'─' * 55}")

    for image_path in sample_images:

        if not os.path.exists(image_path):
            print(f"  WARNING: {image_path} not found — skipping")
            continue

        image    = cv2.imread(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Run full pipeline
        preprocessed = preprocess_image(image)

        # Convert preprocessed back to uint8 for display and metric
        preprocessed_uint8 = (preprocessed * 255).astype('uint8')
        preprocessed_bgr   = cv2.cvtColor(
            preprocessed_uint8, cv2.COLOR_GRAY2BGR
        )

        # Calculate L channel metrics
        before = calculate_l_std_dev(image)
        after  = calculate_l_std_dev(preprocessed_bgr)
        change = after - before

        # Print metrics
        print(f"\n  Sample: {filename}")
        print(f"  {'─' * 42}")
        print(f"  L Std Dev before:  {before:.4f}")
        print(f"  L Std Dev after:   {after:.4f}")
        print(f"  Change:            {change:+.4f}")
        print(f"  Result: {'Contrast IMPROVED ✅' if change > 0 else 'Contrast DECREASED ⚠️'}")

        # Display original vs preprocessed
        show_image(f"{filename} — Original",      resize_image(image))
        show_image(f"{filename} — Preprocessed",  preprocessed_uint8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n{'─' * 55}")
    print(f"  Verification Complete")
    print(f"{'─' * 55}")

# ── Directory Verification ─────────────────────────────────────────────────

def verify_directory_setup():
    """
    Verifies all required input and output directories exist.
    Reports image counts per split before running pipeline.
    Returns True if all checks pass, False otherwise.
    """
    PREPROC_BASE = os.path.join(BASE_DIR, "dataset", "preprocessed")

    input_roots = [
        (TRAIN_DIR,  "Train"),
        (TEST_CLEAN, "Test Clean"),
        (TEST_AUG,   "Test Augmented"),
    ]

    output_roots = [
        os.path.join(PREPROC_BASE, "train"),
        os.path.join(PREPROC_BASE, "test", "clean"),
        os.path.join(PREPROC_BASE, "test", "augmented"),
    ]

    print(f"\n{'─' * 55}")
    print(f"  Directory Verification")
    print(f"{'─' * 55}")

    total_images = 0

    for input_root, label in input_roots:
        split_total = 0
        print(f"\n  Input — {label}")
        print(f"  {input_root}")

        if not os.path.exists(input_root):
            print(f"  ERROR: Directory not found ❌")
            continue

        for denomination in DENOMINATIONS:
            for design in DESIGNS:
                for side in SIDES:
                    folder = os.path.join(
                        input_root, denomination, design, side
                    )
                    if not os.path.exists(folder):
                        print(f"  MISSING: {denomination}/{design}/{side} ❌")
                        continue
                    count = len([
                        f for f in os.listdir(folder)
                        if f.lower().endswith('.jpg')
                    ])
                    split_total += count

        print(f"  Images found: {split_total}")
        total_images += split_total

    print(f"\n  Total input images: {total_images}")
    print(f"\n  Output directories:")

    all_outputs_exist = True
    for output_root in output_roots:
        exists = os.path.exists(output_root)
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status} {output_root}")
        if not exists:
            all_outputs_exist = False

    print(f"\n{'─' * 55}")
    if total_images == 3400 and all_outputs_exist:
        print(f"  Verification PASSED ✅")
        print(f"  Ready to run preprocessing pipeline")
    else:
        print(f"  Verification FAILED ❌")
        if total_images != 3400:
            print(f"  Expected 3400 images — found {total_images}")
        if not all_outputs_exist:
            print(f"  Missing output directories — run setup_project.py")
    print(f"{'─' * 55}")

    return total_images == 3400 and all_outputs_exist

# ── Full Pipeline Runner ───────────────────────────────────────────────────

def run_preprocessing_pipeline():
    """
    Runs full preprocessing pipeline across all three dataset splits.
    Reads from:  dataset/processed/
    Writes to:   dataset/preprocessed/
    """
    PREPROC_BASE = os.path.join(BASE_DIR, "dataset", "preprocessed")

    splits = [
        (TRAIN_DIR,  os.path.join(PREPROC_BASE, "train"),
         "Train"),
        (TEST_CLEAN, os.path.join(PREPROC_BASE, "test", "clean"),
         "Test Clean"),
        (TEST_AUG,   os.path.join(PREPROC_BASE, "test", "augmented"),
         "Test Augmented"),
    ]

    grand_total  = 0
    grand_errors = 0

    print(f"\n{'─' * 55}")
    print(f"  Preprocessing Pipeline Starting")
    print(f"{'─' * 55}")

    for input_root, output_root, label in splits:

        split_total  = 0
        split_errors = 0

        print(f"\n── {label} ──")

        for denomination in DENOMINATIONS:
            for design in DESIGNS:
                for side in SIDES:

                    input_dir  = os.path.join(
                        input_root, denomination, design, side
                    )
                    output_dir = os.path.join(
                        output_root, denomination, design, side
                    )

                    if not os.path.exists(input_dir):
                        continue

                    images = sorted([
                        f for f in os.listdir(input_dir)
                        if f.lower().endswith('.jpg')
                    ])

                    if not images:
                        continue

                    os.makedirs(output_dir, exist_ok=True)

                    for filename in images:
                        input_path  = os.path.join(input_dir, filename)
                        output_path = os.path.join(output_dir, filename)

                        image = cv2.imread(input_path)

                        if image is None:
                            print(f"  ERROR reading: {filename}")
                            split_errors += 1
                            continue

                        preprocessed = preprocess_image(image)
                        save_ready   = (preprocessed * 255).astype('uint8')
                        cv2.imwrite(output_path, save_ready)
                        split_total += 1

        print(f"  Processed: {split_total} images")
        if split_errors > 0:
            print(f"  Errors:    {split_errors} images")

        grand_total  += split_total
        grand_errors += split_errors

    print(f"\n{'─' * 55}")
    print(f"  Preprocessing Pipeline Complete")
    print(f"{'─' * 55}")
    print(f"  Total processed:  {grand_total}")
    print(f"  Total errors:     {grand_errors}")
    print(f"{'─' * 55}")

# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Quality verification — uncomment to verify preprocessing on samples
    # run_comparisons()

    # Directory check — recommended to keep active
    ready = verify_directory_setup()

    if ready:
        print(f"\n  Starting pipeline in 3 seconds...")
        time.sleep(3)
        run_preprocessing_pipeline()
    else:
        print(f"\n  Fix directory issues before running pipeline.")