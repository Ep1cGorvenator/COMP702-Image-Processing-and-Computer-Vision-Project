import cv2
import numpy as np

def segment_image(gray_img, method="canny"):
    """
    Applies chosen segmentation, finds contours, and crops the regions 
    from the original grayscale image.
    """
    if method == "adaptive":
        mask = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 199, 5
        )
    
    elif method == "otsu":
        _, mask = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
    elif method == "canny":
        # 1. Blur first! Canny is highly sensitive to paper grain noise.
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # 2. Apply Canny Edge Detection (50=lower threshold, 150=upper threshold)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 3. Morphological Closing to fuse the disconnected edge lines into solid shapes
        # We use a fairly large 15x15 rectangle kernel to ensure gaps are bridged
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ── Common Contour and Extraction Logic ──
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out tiny dust specs
        if cv2.contourArea(contour) > 750:
            # CRITICAL RULE: Always crop the final pixel data from the ORIGINAL gray_img, 
            # NEVER from the Canny mask or the Threshold mask!
            region = gray_img[y:y+h, x:x+w]
            regions.append(np.asarray(region))
            
    return regions, mask # Returning mask temporarily so you can visually test it